from utils import *
import torch
import torch.nn as nn
import torch.linalg as lin

class NLLsmooth(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(NLLsmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1) #along last direction
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)) #increase the dimension
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def loss_fn_kd(outputs, labels, teacher_outputs,args):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = args.alpha
    T = args.T
    norm_loss = lin.norm(outputs.detach(),dim=1).mean() * args.norm_loss
    KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
        kerem_softmax(teacher_outputs/T)) * (alpha * T * T)
    CE = F.cross_entropy(outputs, labels) * (1. - alpha)
    loss_dic = {'CE':CE,'KL':KL,'Norm':norm_loss}
    return KL + CE + norm_loss,loss_dic

def loss_trades(adv_logits,adv_lat,clean_logits,clean_lat,labels,buffer,args):
    KLD = nn.KLDivLoss(size_average=False)
    bs = adv_logits.size(0)
    natural_ce = F.cross_entropy(clean_logits,labels)
    robust_kld = KLD(F.log_softmax(adv_logits, dim=1),F.softmax(clean_logits, dim=1)) / bs
    cos_clean = cosine_sim_logitz_buffered(clean_lat,labels,buffer)
    cos_adv = cosine_sim_logitz_buffered(adv_lat,labels,buffer)
    buffer_clean_kl = KLD(F.log_softmax(clean_logits, dim=1),kerem_softmax(cos_clean)) / bs
    buffer_adv_kl = KLD(F.log_softmax(adv_logits, dim=1),kerem_softmax(cos_adv)) / bs
    loss_dic = {'CE':natural_ce.item(),'KL1':buffer_clean_kl.item(),'KL2':robust_kld.item(),'KL3':buffer_adv_kl.item()}
    total_loss =  natural_ce * (1-args.alpha) + buffer_clean_kl * args.alpha + robust_kld * args.Beta_1 + buffer_adv_kl  * args.Beta_2
    return total_loss, loss_dic

def apr2(adv_logits,adv_lat,clean_logits,clean_lat,labels,buffer,args):
    KLD = nn.KLDivLoss(size_average=False)
    bs = adv_logits.size(0)
    natural_ce = F.cross_entropy(clean_logits,labels)
    robust_ce = F.cross_entropy(adv_logits,labels)
    cos_clean = cosine_sim_logitz_buffered(clean_lat,labels,buffer)
    cos_adv = cosine_sim_logitz_buffered(adv_lat,labels,buffer)
    buffer_clean_kl = KLD(F.log_softmax(clean_logits, dim=1),kerem_softmax(cos_clean)) / bs
    buffer_adv_kl = KLD(F.log_softmax(adv_logits, dim=1),kerem_softmax(cos_adv)) / bs
    loss_dic = {'CE1':natural_ce.item(),'CE2':robust_ce.item(),'KL1':buffer_clean_kl.item(),'KL2':buffer_adv_kl.item()}
    total_loss =  natural_ce * (1-args.alpha) + robust_ce * (1-args.alpha) + buffer_clean_kl * args.alpha + buffer_adv_kl * args.alpha
    return total_loss, loss_dic

def apr3(adv_logits,clean_logits,counter_logits,counter_lat,labels,buffer,args):
    KLD = nn.KLDivLoss(size_average=False)
    bs = adv_logits.size(0)
    counter_CE = F.cross_entropy(counter_logits,labels)
    cos_cadv = cosine_sim_logitz_buffered(counter_lat,labels,buffer)
    buffer_cadv_kl = KLD(F.log_softmax(counter_logits, dim=1),kerem_softmax(cos_cadv)) / bs
    robust_counter_kld = KLD(F.log_softmax(adv_logits, dim=1), F.softmax(counter_logits, dim=1)) / bs
    clean_kld = KLD(F.log_softmax(counter_logits, dim=1), F.softmax(clean_logits, dim=1)) / bs
    loss_dic = {'CE':counter_CE.item(),'KL1':buffer_cadv_kl.item(),'KL2':robust_counter_kld.item(),'KL3':clean_kld.item()}
    total_loss = counter_CE * (1-args.alpha) + buffer_cadv_kl * args.alpha + robust_counter_kld + clean_kld
    return total_loss, loss_dic