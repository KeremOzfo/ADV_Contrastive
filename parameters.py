import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help='gpu_id')
    parser.add_argument('--num_trail', type=int, default=3, help='number of trail')
    parser.add_argument('--epoch', type=int, default=120, help='number of Epochs for train')
    parser.add_argument('--LR_change', type=list, default=[75,90], help='Lr drops @')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='cifar10,cifar100,imagenet')
    parser.add_argument('--nn_name', type=str, default='resnet18', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--bs', type=int, default=128, help='Mini-batch Size')
    parser.add_argument("--include", type=str, default='-', help='layer to include for model update')
    parser.add_argument("--loss", type=str, default='KLD', help='KLD or CE')

    parser.add_argument("--pgd_train_iter", type=int, default=10, help='# of pgd iterations during training')
    parser.add_argument("--pgd_test_iter", type=int, default=20, help='# of pgd iterations during test')
    parser.add_argument("--pgd_train_alpha", type=float, default=2.0/255, help='alpha parameter')
    parser.add_argument("--pgd_test_alpha", type=float, default=2.0/255, help='alpha parameter')
    parser.add_argument("--pgd_train_epsilon", type=float, default=8.0/255.0, help='epsilon')
    parser.add_argument("--pgd_test_epsilon", type=float, default=8.0/255.0, help='epsilon')

    parser.add_argument('--alpha', type=float, default=0.1, help='alpha parameter for KLD Loss')
    parser.add_argument('--Beta_1', type=float, default=6, help='alpha parameter for KLD Loss')
    parser.add_argument('--Beta_2', type=float, default=0.5, help='alpha parameter for KLD Loss')
    parser.add_argument('--T', type=int, default=5, help='Temperature parameter for KLD loss')
    parser.add_argument('--latent_buffer', type=bool, default=True, help='store latent values on a buffer')
    parser.add_argument('--buffer_size', type=int, default=16, help='buffer size per class')


    parser.add_argument("--save_model", type=bool, default=True, help='save the trained model')
    parser.add_argument("--AA_model", type=str, default='ADV_cons_model-T_5-Alpha_0.1.pt', help='save the trained model')

    args = parser.parse_args()
    return args

def log_params(args,f_acc_adv,f_acc_clean,loc,trail=0):
    f = open(loc + '/simulation_Details.txt', 'a+')
    if trail==0:
        f.write('############## Args ###############' + '\n')
        for arg in vars(args):
            line = str(arg) + ' : ' + str(getattr(args, arg))
            f.write(line + '\n')
        f.write('############ Results ###############' + '\n')
    if trail< args.num_trail:
        f.write('Trail {} Adv_acc: {} , Clean accuracy: {} \n'.format(trail,f_acc_adv,f_acc_clean))
    else:
        f.write('AVG adv acc: {}, AVG clean accuracy: {}'.format(f_acc_adv,f_acc_clean))
    f.close()

def get_path(args):
    path = 'Results'
    if not os.path.exists(path):
        os.mkdir(path)
    if args.loss == 'KLD':
        file = 'AdvDistill-Alpha_{}-Temp_{}-dataset_{}'.format(args.alpha, args.T,args.dataset_name)
    else:
        file = 'Benchmark-include_{}-dataset_{}'.format(args.include,args.dataset_name)
    path = os.path.join(path, file)
    while os.path.exists(path):
        path += '+'
    return path