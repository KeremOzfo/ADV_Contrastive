from nn_classes import get_net
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from dataset import *
from parameters import *
from utils import *
import numpy as np
import torch.nn.functional as F
from train_funcs import *

def save_model(args,model,type):
    if args.save_model:
        if not os.path.exists('Saved_models'):
            os.mkdir('Saved_models')
        if args.loss == 'KLD':
            f_name = '{}-ADV_cons_model-T_{}-Alpha_{}.pt'.format(type,args.T,args.alpha)
        else:
            f_name = '{}-ADV_model-eps_{}.pt'.format(type,args.pgd_train_epsilon*255)
        pat = os.path.join('Saved_models',f_name)
        torch.save(model.state_dict(), pat)


if __name__ == '__main__':
    args = args_parser()
    path = get_path(args)
    os.mkdir(path)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loader(args)
    model = get_net(args).to(device)
    dictz = torch.load('{}_{}_pretrained_dict.pt'.format(args.nn_name, args.dataset_name), map_location=device)
    # dictz = torch.load('Saved_models/ADV_model-eps_2.0.pt', map_location=device)
    model.load_state_dict(dictz)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=5e-4)
    adv_accs = []
    clean_accs = []
    schedular = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=args.LR_change,gamma=0.1)
    attack = utils.pgd_linf_untargeted
    buffer = Buffer(args,device) if args.latent_buffer else None
    for ep in tqdm(range(args.epoch)):
        train_err = epoch(args,train_loader, model,buffer=buffer,attack=attack, opt=opt,device=device,num_iter=args.pgd_train_iter,
                                                                         epsilon=args.pgd_train_epsilon, randomize=True,
                                                                         alpha=args.pgd_train_alpha)
        adv_test_acc = epoch(args,test_loader,model,attack=attack,device=device,num_iter=args.pgd_test_iter,
                                                                         epsilon=args.pgd_test_epsilon, randomize=True,
                                                                         alpha=args.pgd_test_alpha)
        clean_test_acc = epoch(args,test_loader,model,device=device,num_iter=args.pgd_test_iter,
                                                                         epsilon=args.pgd_test_epsilon, randomize=True,
                                                                         alpha=args.pgd_test_alpha)
        schedular.step()
        adv_acc = adv_test_acc * 100
        clean_acc = clean_test_acc * 100
        adv_accs.append(adv_acc)
        clean_accs.append(clean_acc)
        if adv_acc >= max(adv_accs):
            save_model(args, model, 'best')
        print('Adv acc: ',adv_acc, 'Clean acc: ', clean_acc)
    log_params(args, adv_accs[-1],clean_accs[-1], path)
    adv_save_path = os.path.join(path, 'adv_result-{}'.format(0))
    clean_save_path = os.path.join(path, 'clean_result-{}'.format(0))
    np.save(adv_save_path, adv_accs)
    np.save(clean_save_path, clean_accs)
    save_model(args,model,'last')