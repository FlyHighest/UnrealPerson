import os, torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.resnet import resnet50
from utils.dataset import imgdataset, imgdataset_camtrans, IdentityCameraSampler, NormalCollateFn
from utils.losses import Losses#, LocalLoss, GlobalLoss
from utils.evaluators import evaluate_all
from utils.lr_adjust import StepLrUpdater, SetLr
from utils.util import CamDataParallel , organize_data
from utils.logger import Logger
import sys
import argparse
from tqdm import tqdm,trange
import time
from getpass import getuser

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--src',default='market,duke,msmt17,unreal',type=str, help='src domaim')
parser.add_argument('--tar',default='market,duke',type=str, help='target domain')
parser.add_argument('--max_ep', default=100, type=int, help='maximum epoch number')

parser.add_argument('--num_cam', default=8, type=int, help='target camera number')

opt = parser.parse_args()

t_start = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
###########   HYPER   ###########
base_lr = 0.01
num_epoches = 100
batch_size = 128
num_instances=4
K = 4
num_cam = opt.num_cam
sys.stdout = Logger(os.path.join('./snapshot',opt.name,'log_train.txt'))
##########   DATASET   ###########
dataset_path = 'data/' 
if opt.src=='unreal':
    src_dir = dataset_path
else:
    src_dir = dataset_path +opt.src+ '/bounding_box_train_camstyle_merge/'
tar_dir = dataset_path + opt.tar+ '/bounding_box_train_camstyle_merge/'
tar_dir_test = dataset_path +opt.tar+ '/'

src_annfile = 'list_{}/list_{}_train.txt'.format(opt.src,opt.src)
tar_annfile = 'list_{}/list_{}_train.txt'.format(opt.tar,opt.tar)
tar_annfile_test = 'list_{}/list_{}_test.txt'.format(opt.tar,opt.tar)

#resnet50: https://download.pytorch.org/models/resnet50-19c8e357.pth
imageNet_pretrain = '/home/'+getuser()+'/.torch/models/resnet50-19c8e357.pth'


train_dataset = imgdataset(dataset_dir=src_dir, txt_path=src_annfile, transformer='train')

sampler = IdentityCameraSampler(train_dataset.data_source, batch_size, num_instances)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler = sampler, num_workers=4, drop_last=True, collate_fn = NormalCollateFn())

train_dataset_t = imgdataset_camtrans(dataset_dir=tar_dir, txt_path=tar_annfile, 
    transformer='train', num_cam=num_cam, K=K)
train_loader_t = DataLoader(dataset=train_dataset_t, batch_size=int(batch_size/K), shuffle=True, num_workers=4, drop_last=True)


###########   MODEL   ###########
model, param = resnet50(pretrained=imageNet_pretrain, num_classes=len(train_dataset.pids))
model.cuda()
model = CamDataParallel(model)#, device_ids=[0,1])

losses = Losses(K=K, 
    batch_size=batch_size, 
    bank_size=len(train_dataset_t), 
    ann_file=tar_annfile, 
    cam_num=num_cam)
losses = losses.cuda()
optimizer = torch.optim.SGD(param, lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

###########   TRAIN   ###########
target_iter = iter(train_loader_t)
for epoch in trange(1, num_epoches+1):

    lr = StepLrUpdater(epoch, base_lr=base_lr, gamma=0.1, step=40)
    SetLr(lr, optimizer)

    print('-' * 10)
    print('Epoch [%d/%d], lr:%f'%(epoch, num_epoches, lr))

    running_loss_src = 0.0
    running_loss_local = 0.0
    running_loss_global = 0.0

    if (epoch)%5 == 0 and epoch!=opt.max_ep:
        losses.reset_multi_label(epoch)

    model.train()
    for i, source_data in enumerate(train_loader, 1):
        try:
            target_data = next(target_iter)
        except:
            target_iter = iter(train_loader_t)
            target_data = next(target_iter)
        image_src = source_data[0].cuda()
        label_src = source_data[1].cuda()
        image_tar = target_data[0].cuda()    
        image_tar = image_tar.view(-1, image_tar.size(2), image_tar.size(3), image_tar.size(4))
        label_tar = target_data[2].cuda()
        cams_src = source_data[-1].cuda()
        cams_tar = target_data[-1].cuda().view(-1)
        label_tar = label_tar.view(-1)
        image_src_org, label_src_org = organize_data(image_src, cams_src , label_src)
        image_tar_org, label_tar_org = organize_data(image_tar, cams_tar , label_tar)
        x_src = model(image_src_org)[0]
        x_tar = model(image_tar_org)[2]
        loss_all= losses(x_src, label_src_org, x_tar, label_tar_org, epoch)
        loss, loss_s, loss_l, loss_g = loss_all


        running_loss_src += loss_s.mean().item()
        running_loss_local += loss_l.mean().item()
        running_loss_global += loss_g.mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update_memory(x_tar, label_tar_org, epoch=epoch)
       
        if i % 50 == 0:
            print('  iter: %3d/%d,  loss src: %.3f, loss local: %.3f, loss global: %.3f'%(i, len(train_loader), running_loss_src/i, running_loss_local/i, running_loss_global/i))
        if i==300:
            break 

    print('Finish {} epoch\n'.format(epoch))

    if epoch % 10 ==0 and epoch>45:
        if hasattr(model, 'module'):
            model_save = model.module
        else:
            model_save = model
        torch.save(model_save.state_dict(), 'snapshot/{}/resnet50_{}2{}_epoch{}_cbn.pth'.format(opt.name,opt.src,opt.tar,epoch))    
        
    if epoch >= opt.max_ep:
        print('Training stops at {}. Total Time: {} minutes.'.format(epoch,(time.time()-t_start)//60))
        break
