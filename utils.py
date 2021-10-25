import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.linalg as lin
from torch.autograd import Variable
import matplotlib.pyplot as plt
from os.path import join

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


def counter_pgd_linf_untargeted_L1(model, X, eps_ball=0.01,epsilon=0.1, alpha=0.01, num_iter=10, randomize=False,start_delta=None):
    if randomize:
        delta_counter = torch.rand_like(X, requires_grad=True)
        delta_counter.data = delta_counter.data * 2 * epsilon - epsilon
    else:
        delta_counter = torch.zeros_like(X, requires_grad=True)
        if start_delta is not None:
            delta_counter.data = start_delta.data

    #predict, latent = model(X)
    for t in range(num_iter):
        predict_adv, latent_adv = model(X + delta_counter)
        loss = torch.mean(lin.norm(latent_adv,ord=1,dim=1))
        loss.backward()

        img_grads = torch.clone(delta_counter.grad.detach())
        sign_vec = img_grads.sign()

        delta_counter.data = (delta_counter - alpha * sign_vec).clamp(-epsilon, epsilon)
        delta_counter.data = start_delta.data + (start_delta.data - delta_counter).clamp(-eps_ball, eps_ball)
        delta_counter.data = (X.data + delta_counter).clamp(0, 1) - X.data
        delta_counter.grad.zero_()
    return delta_counter.detach()

def counter_train(model, X, y,device,epsilon=0.1,eps_ball=0.01,alpha=0.01, num_iter=10, randomize=False,start_delta=None):
    if randomize:
        delta_counter = torch.rand_like(X, requires_grad=True,device=device)
        delta_counter.data = delta_counter.data * 2 * epsilon - epsilon
    else:
        delta_counter = torch.zeros_like(X, requires_grad=True,device=device)
        if start_delta is not None:
            delta_counter.data = start_delta.data

    #predict, latent = model(X)
    for t in range(num_iter):
        predict_adv, latent_adv = model(X + delta_counter)
        loss = torch.mean(lin.norm(latent_adv,ord=1,dim=1)) + F.cross_entropy(predict_adv,y)
        #print(torch.mean(lin.norm(latent_adv,ord=1,dim=1)),F.cross_entropy(predict_adv,y))
        loss.backward()

        img_grads = torch.clone(delta_counter.grad.detach())
        sign_vec = img_grads.sign()

        delta_counter.data = (delta_counter - alpha * sign_vec).clamp(-epsilon, epsilon)
        delta_counter.data = start_delta.data + (start_delta.data - delta_counter).clamp(-eps_ball, eps_ball)
        delta_counter.data = (X.data + delta_counter).clamp(0, 1) - X.data
        delta_counter.grad.zero_()
    return delta_counter.detach()

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


def graph_loss(dic,args,path):
    x_= range(args.epoch)
    for loss_type in dic:
        plt.plot(x_, dic[loss_type], label=loss_type)
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.legend()
    plt.title('{}'.format(args.loss))
    plt.grid(True)
    plt.tight_layout()
    name= path.split('/')[-1] +'LossPlot.png'
    save = join(path,name)
    plt.savefig(save,bbox_inches='tight')

