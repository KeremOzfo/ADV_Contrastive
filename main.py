import numpy as np

from nn_classes import get_net, get_resnet18_AA
import torch
import torch.optim as optim
from tqdm import tqdm
from dataset import *
from parameters import *
from train_funcs import *
from AA_test import AA_epoch
from copy import deepcopy

def save_model(args,model,type):
    if args.save_model:
        if not os.path.exists('Saved_models'):
            os.mkdir('Saved_models')
        if args.loss == 'KLD':
            f_name = '{}-ADV_cons_model-T_{}-Alpha_{}.pt'.format(type,args.T,args.alpha)
        elif args.loss =='trades':
            f_name = '{}-ADV_trades_model-Alpha_{}_Beta1_{}-Beta2_{}.pt'.format(type, args.alpha,args.Beta_1,args.Beta_2)
        elif args.loss =='apr2':
            f_name = '{}-ADV_apr2_model-Alpha_{}.pt'.format(type, args.alpha)
        elif args.loss =='apr3':
            f_name = '{}-ADV_apr3_model-Alpha_{}.pt'.format(type, args.alpha)
        else:
            f_name = '{}-ADV_model.pt'.format(type)
        pat = os.path.join('Saved_models',f_name)
        torch.save(model.state_dict(), pat)
        return pat


if __name__ == '__main__':
    args = args_parser()
    best_loc = None
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loader(args)
    model = get_net(args).to(device)
    dictz = torch.load('{}_{}_pretrained_dict.pt'.format(args.nn_name, args.dataset_name), map_location=device)
    # dictz = torch.load('Saved_models/ADV_model-eps_2.0.pt', map_location=device)
    model.load_state_dict(dictz)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=5e-4)
    adv_accs = []
    clean_accs = []
    losses =[]
    loss_dicts = None
    schedular = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=args.LR_change,gamma=0.1)
    attack = utils.pgd_linf_untargeted
    buffer = Buffer(args,device) if args.latent_buffer else None
    for ep in tqdm(range(args.epoch)):
        train_err,train_loss,loss_dict = epoch(args,train_loader, model,buffer=buffer,attack=attack, opt=opt,device=device,num_iter=args.pgd_train_iter,
                                                                         epsilon=args.pgd_train_epsilon, randomize=True,
                                                                         alpha=args.pgd_train_alpha)
        adv_test_acc,_,_ = epoch(args,test_loader,model,attack=attack,device=device,num_iter=args.pgd_test_iter,
                                                                         epsilon=args.pgd_test_epsilon, randomize=True,
                                                                         alpha=args.pgd_test_alpha)
        clean_test_acc,_,_ = epoch(args,test_loader,model,device=device,num_iter=args.pgd_test_iter,
                                                                         epsilon=args.pgd_test_epsilon, randomize=True,
                                                                         alpha=args.pgd_test_alpha)
        schedular.step()
        adv_acc = adv_test_acc * 100
        clean_acc = clean_test_acc * 100
        losses.append(train_loss)
        adv_accs.append(adv_acc)
        clean_accs.append(clean_acc)
        if loss_dict is not None:
            if ep==0:
                loss_dicts = {k: [] for k in loss_dict}
            [loss_dicts[key].append(loss_dict[key]) for key in loss_dict]
        if adv_acc >= max(adv_accs):
            best_loc = save_model(args, model, 'best')
        print('Adv acc: ',adv_acc, 'Clean acc: ', clean_acc)
    path = get_path(args)
    os.mkdir(path)
    log_params(args, adv_accs[-1],clean_accs[-1], path)
    adv_save_path = os.path.join(path, 'adv_result-{}'.format(0))
    clean_save_path = os.path.join(path, 'clean_result-{}'.format(0))
    loss_path = os.path.join(path, 'loss_result-{}'.format(0))
    np.save(adv_save_path, adv_accs)
    np.save(clean_save_path, clean_accs)
    np.save(loss_path,losses)
    last_loc = save_model(args,model,'last')
    if loss_dicts is not None:
        graph_loss(loss_dicts,args,path)

    ## Auto Attack
    model_aa = get_resnet18_AA().to(device)
    model_aa.load_state_dict(model.state_dict())
    AA_last = AA_epoch(args,test_loader,model_aa,device)
    best_dict = torch.load(best_loc, map_location=device)
    model_aa.load_state_dict(best_dict)
    AA_best = AA_epoch(args,test_loader,model_aa,device)
    log_AA(AA_last,AA_best,path)