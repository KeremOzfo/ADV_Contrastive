import time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from loss_funcs import *

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
        if opt:
            opt.zero_grad()
            predicts, latents = model(img + delta)
            if buffer is None or args.loss=='CE':
                loss = nn.CrossEntropyLoss()(predicts,label)
            else:
                if buffer.buffer_full:
                    if args.loss =='KLD':
                        loss, loss_dic = loss_fn_kd(predicts, label, cosine_sim_logitz_buffered(latents,label,buffer),args)
                    elif args.loss =='trades':
                        clean_predicts, clean_latents = model(img)
                        loss, loss_dic = loss_trades(predicts,latents,clean_predicts,
                                                     clean_latents,label,buffer,args)
                    elif args.loss =='apr2':
                        clean_predicts, clean_latents = model(img)
                        loss, loss_dic = apr2(predicts, latents, clean_predicts,
                                                     clean_latents, label, buffer, args)
                    elif args.loss =='apr3':
                        counter_delta = counter_pgd_linf_untargeted_L1(model,img,epsilon=args.pgd_counter_epsilon,
                                                                   alpha=args.pgd_counter_alpha,start_delta=delta)
                        counter_clamp = delta.sub(counter_delta).clamp(-1/255,1/255)
                        counter_predicts, counter_latents = model((img+counter_clamp).clamp(0,1))
                        clean_predicts, clean_latents = model(img)
                        loss, loss_dic = apr3(predicts, clean_predicts, counter_predicts,
                                                     counter_latents, label, buffer, args)
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
        else:
            predicts, latents = model(img + delta)
        total_acc += (predicts.max(dim=1)[1] == label).sum().item()
    if opt:
        print('Loss: ', total_loss / len(loader.dataset))
        if losses is not None:
            st = ''
            for key in losses:
                losses[key] /= len(loader.dataset)
                st += '{} {}, '.format(key,losses[key])
            print(st)
    return total_acc / len(loader.dataset), total_loss/len(loader.dataset),losses



def get_cleans(model,data):
    model.eval()
    predicts, latents = model(data)
    model.train()
    return predicts,latents.detach()