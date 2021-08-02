import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.linalg as lin
from torch.autograd import Variable


class Buffer():
    '''
    Stores latent values of the recent batches.
    '''

    def __init__(self,args,device):
        super(Buffer, self).__init__()
        self.buffer_size = args.buffer_size
        self.buffer_full = False
        if args.dataset_name == 'cifar10':
            self.num_cls = 10
        elif args.dataset_name == 'cifar100':
            self.num_cls = 100
        elif args.dataset_name == 'imagenet':
            self.num_cls = 200

        self.buffer = torch.zeros((self.num_cls,self.buffer_size,512),device=device)
        self.buffer_labels = torch.zeros(self.num_cls*self.buffer_size,device=device)
        self.buffer_map = torch.zeros(self.num_cls,device=device)
        for cls in range(self.num_cls):
            self.buffer_labels[self.buffer_size*cls:self.buffer_size*(cls+1)] = cls

    def get_buffer_2d(self):
        return self.buffer.view(-1,512)

    def update_buffer(self,new_latents,labels):
        lower, upper = torch.min(labels).int(), torch.max(labels).int()
        for cls in range(lower,upper+1):
            cls_mask = labels == cls
            cls_latents = new_latents[cls_mask]
            buffer_lat = torch.clone(self.buffer[cls].flatten())
            buffer_lat = torch.cat((cls_latents.flatten(),buffer_lat),dim=0)[:self.buffer_size*512]
            self.buffer[cls] = buffer_lat.view(self.buffer[cls].size())
            if not self.buffer_full:
                self.buffer_map[cls] += cls_mask.sum()
        if self.buffer_map.min().ge(self.buffer_size) and not self.buffer_full:
            self.buffer_full = True

def pgd_linf_untargeted(model, X, y, device, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True,device=device)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True,device=device)
    for t in range(num_iter):
        predict,latent = model(X + delta)
        loss = nn.CrossEntropyLoss()(predict, y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.data = (X.data + delta).clamp(0,1) - X.data
        delta.grad.zero_()
    return delta.detach()


def cw_whitebox(model,X,y,epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for _ in range(num_iter):

        output, l = model(X + delta)
        correct_logit = torch.sum(torch.gather(output, 1, (y.unsqueeze(1)).long()).squeeze())
        tmp1 = torch.argsort(output, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        wrong_logit = torch.sum(torch.gather(output, 1, (new_y.unsqueeze(1)).long()).squeeze())
        loss = -F.relu(correct_logit - wrong_logit)

        loss.backward()
        delta_sign = alpha * delta.grad.data.detach().sign()
        delta.data = torch.clamp(delta + delta_sign, -epsilon, epsilon) ## add or sub delta_sign ?
        delta.data = (X.data + delta).clamp(0, 1) - X.data
        delta.grad.zero_()
    return delta

def sim_matrix2(a,b,eps=1e-8):
    a_n ,b_n = a.norm(dim=1)[:, None] , b.norm(dim=1)[:,None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def cosine_sim_logitz_buffered(logitz,labels,buffer):
    buffer2d = buffer.get_buffer_2d()
    num_cls, buffer_size, lat_size = buffer.buffer.size()
    sim_mt = sim_matrix2(logitz,buffer2d)
    avg_cos = sim_mt.view(labels.size(0),num_cls,buffer_size).mean(dim=2)
    return avg_cos

def kerem_softmax(logit):
    soft = logit.exp() / logit.exp().sum(dim=1,keepdim=True).expand(logit.size())
    return soft

def loss_fn_kd(outputs, labels, teacher_outputs,args):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = args.alpha
    T = args.T
    norm_loss = lin.norm(outputs.detach(),dim=1).mean()
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
        kerem_softmax(teacher_outputs/T)) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss + norm_loss * args.norm_loss

def loss_trades(adv_logits,adv_lat,clean_logits,clean_lat,labels,buffer,args):
    KLD = nn.KLDivLoss(size_average=False)
    bs = adv_logits.size(0)
    natural_ce = F.cross_entropy(clean_logits,labels) * (1-args.alpha)
    robust_kld = KLD(F.log_softmax(adv_logits, dim=1),F.softmax(clean_logits, dim=1)) * args.Beta_1 / bs
    cos_clean = cosine_sim_logitz_buffered(clean_lat,labels,buffer)
    cos_adv = cosine_sim_logitz_buffered(adv_lat,labels,buffer)
    buffer_clean_kl = KLD(F.log_softmax(clean_logits, dim=1),kerem_softmax(cos_clean)) * args.alpha / bs
    buffer_adv_kl = KLD(F.log_softmax(adv_logits, dim=1),kerem_softmax(cos_adv)) * args.Beta_2 / bs
    #loss_dic = {'CE':natural_ce.item(),'KL1':0,'KL2':robust_kld.item(),'KL3':0}
    loss_dic = {'CE':natural_ce.item(),'KL1':buffer_clean_kl.item(),'KL2':robust_kld.item(),'KL3':buffer_adv_kl.item()}
    total_loss =  natural_ce + buffer_clean_kl + robust_kld + buffer_adv_kl
    #total_loss = natural_ce + robust_kld
    return total_loss, loss_dic


def include_fc(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
            layer.weight.grad.zero_()
            if layer.bias is not None:
                layer.bias.grad.zero_()

def include_bn(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.grad.zero_()
            if layer.bias is not None:
                layer.bias.grad.zero_()

def include_fc_bn(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.weight.grad.zero_()
            if layer.bias is not None:
                layer.bias.grad.zero_()

def grad_nonzero(model, args):
    if args.include == 'fc':
        include_fc(model)
    elif args.include == 'bn':
        include_bn(model)
    elif args.include == 'fc-bn' or args.include == 'bn-fc':
        include_fc_bn(model)