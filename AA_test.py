from autoattack import AutoAttack
from nn_classes import *
from parameters import *
from dataset import get_data_loader



def AA_epoch(args,loader, model, device=None):
    total_acc, total_loss= 0., 0.
    eps = args.pgd_test_epsilon
    adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard',device=device)
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        x_adv = adversary.run_standard_evaluation(img, label, bs=args.bs)
        predicts = model(x_adv)
        total_acc += (predicts.max(dim=1)[1] == label).sum().item()
    return total_acc / len(loader.dataset)

if __name__ == '__main__':
    args = args_parser()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loader(args)
    model = get_resnet18_AA().to(device)
    dictz = torch.load('Saved_models/{}'.format(args.AA_model), map_location=device)
    model.load_state_dict(dictz)

    test_acc = AA_epoch(args,test_loader,model,device)
    print(test_acc)