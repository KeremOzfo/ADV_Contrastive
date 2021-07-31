import time

import torch
from utils import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

def epoch(args,loader, model, attack=None, buffer=None,opt=None, device=None,**kwargs):
    total_acc, total_loss= 0., 0.
    losses = None
    loss_dic = None
    if opt is None:
        model.eval()
    else:
        model.train()
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        delta = torch.zeros_like(img, device=device) if attack is None else attack(model, img, label,device,**kwargs)
        predicts, latents = model(img+delta)
        if opt:
            opt.zero_grad()
            if buffer is None or args.loss=='CE':
                loss = nn.CrossEntropyLoss()(predicts,label)
            else:
                if buffer.buffer_full:
                    if args.loss =='KLD':
                        loss = loss_fn_kd(predicts, label, cosine_sim_logitz_buffered(latents,label,buffer),args)
                    elif args.loss =='trades':
                        clean_predicts, clean_latents = get_cleans(model, img)
                        loss, loss_dic = loss_trades(predicts,latents,clean_predicts,
                                                     clean_latents,label,buffer,args)
                    else:
                        raise NotImplementedError('Incompatible loss function')
                else:
                    loss = nn.CrossEntropyLoss()(predicts, label)
            loss.backward()
            grad_nonzero(model,args)
            opt.step()

            if buffer is not None:
                clean_predicts, clean_latents = get_cleans(model,img)
                buffer.update_buffer(clean_latents, label)
            total_loss += loss.item() * img.size(0)

            if loss_dic is not None:
                if losses is None:
                    losses = deepcopy(loss_dic)
                    for key in losses:
                        losses[key] *= img.size(0)
                else:
                    for key in losses:
                        losses[key] += loss_dic[key] * img.size(0)
        total_acc += (predicts.max(dim=1)[1] == label).sum().item()
    if opt:
        print('Loss: ', total_loss / len(loader.dataset))
        if losses is not None:
            st = ''
            for key in losses:
                st += '{} {}, '.format(key,losses[key] / len(loader.dataset))
            print(st)
    return total_acc / len(loader.dataset)



def get_cleans(model,data):
    model.eval()
    predicts, latents = model(data)
    model.train()
    return predicts,latents.detach()