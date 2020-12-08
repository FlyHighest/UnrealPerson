from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import random
import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import opt

from io_stream import data_manager, NormalCollateFn, IdentitySampler, IdentityCameraSampler

from frameworks.models import ResNetBuilder
from frameworks.training import CameraClsTrainer, get_our_optimizer_strategy, CamDataParallel

from utils.serialization import Logger, save_checkpoint, load_moco_model, load_previous_model
from utils.transforms import TrainTransform
from utils.loss import TripletLoss


def train(**kwargs):
    opt._parse(kwargs)
    # torch.backends.cudnn.deterministic = True  # I think this line may slow down the training process
    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(os.path.join('./pytorch-ckpt/current', opt.save_dir, 'log_train.txt'))

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
    else:
        print('currently using cpu')
    print(opt._state_dict())
    print('initializing dataset {}'.format(opt.trainset_name))
    if opt.trainset_name=='combine':
         #input dataset name as 'datasets'
        train_dataset= data_manager.init_combine_dataset(name=opt.trainset_name,options=opt,
                                                         datasets=opt.datasets,
                                              num_bn_sample=opt.batch_num_bn_estimatation * opt.test_batch,
                                              share_cam=opt.share_cam,num_pids=opt.num_pids)
    elif opt.trainset_name=='unreal':
         # input dataset dir in 'datasets'
        train_dataset = data_manager.init_unreal_dataset(name=opt.trainset_name,
                                                         datasets = opt.datasets, 
                                                         num_pids=opt.num_pids,
                                                         num_cams=opt.num_cams,
                                                         img_per_person = opt.img_per_person)

    else:
        train_dataset = data_manager.init_dataset(name=opt.trainset_name,
                                              num_bn_sample=opt.batch_num_bn_estimatation * opt.test_batch,num_pids=opt.num_pids)
    pin_memory = True if use_gpu else False
    summary_writer = SummaryWriter(os.path.join('./pytorch-ckpt/current', opt.save_dir, 'tensorboard_log'))

    if opt.cam_bal:    
        IDSampler=IdentityCameraSampler
    else:              
        IDSampler=IdentitySampler
    if opt.trainset_name=='combine':
        samp = IDSampler(train_dataset.train, opt.train_batch, opt.num_instances,train_dataset.cams_of_dataset,train_dataset.len_of_real_dataset)
    else:
        samp = IDSampler(train_dataset.train, opt.train_batch, opt.num_instances)

    trainloader = DataLoader(
        data_manager.init_datafolder(opt.trainset_name, train_dataset.train, TrainTransform(opt.height, opt.width)),
        sampler=samp ,
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True, collate_fn=NormalCollateFn()
    )
    print('initializing model ...')
    num_pid = train_dataset.num_train_pids if opt.loss=='softmax' else None
    model = ResNetBuilder(num_pid)
    if opt.model_path is not None and 'moco' in opt.model_path:
        model = load_moco_model(model,opt.model_path)
    elif opt.model_path is not None:
        model = load_previous_model(model, opt.model_path, load_fc_layers=False)
    optim_policy = model.get_optim_policy()
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    if use_gpu:
        model = CamDataParallel(model).cuda()

    xent = nn.CrossEntropyLoss()
    triplet = TripletLoss()
    def standard_cls_criterion(feat,
                               preditions,
                               targets,
                               global_step,
                               summary_writer):
        identity_loss = xent(preditions, targets)
        identity_accuracy = torch.mean((torch.argmax(preditions, dim=1) == targets).float())
        summary_writer.add_scalar('cls_loss', identity_loss.item(), global_step)
        summary_writer.add_scalar('cls_accuracy', identity_accuracy.item(), global_step)
        return identity_loss

    def triplet_criterion(feat,preditons,targets,global_step,summary_writer):
        triplet_loss, acc = triplet(feat,targets)
        summary_writer.add_scalar('loss', triplet_loss.item(), global_step)
        print(np.mean(acc.item()))
        summary_writer.add_scalar('accuracy', acc.item(), global_step)
        return triplet_loss
        


    # get trainer and evaluator
    optimizer, adjust_lr = get_our_optimizer_strategy(opt, optim_policy)
    if opt.loss=='softmax':
        crit = standard_cls_criterion
    elif opt.loss=='triplet':
        crit = triplet_criterion
    reid_trainer = CameraClsTrainer(opt, model, optimizer, crit, summary_writer)

    print('Start training')
    for epoch in range(opt.max_epoch):
        adjust_lr(optimizer, epoch)
        reid_trainer.train(epoch, trainloader)
        if (epoch+1)%opt.save_step==0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
        
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
            }, save_dir=os.path.join('./pytorch-ckpt/current', opt.save_dir), ep=epoch+1)

       # if (epoch+1)%15==0:
       #     save_checkpoint({
       #         'state_dict': state_dict,
       #         'epoch': epoch + 1,
       #         }, save_dir=os.path.join('./pytorch-ckpt/current', opt.save_dir))

    if use_gpu:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    save_checkpoint({
        'state_dict': state_dict,
        'epoch': epoch + 1,
    }, save_dir=os.path.join('./pytorch-ckpt/current', opt.save_dir))


if __name__ == '__main__':
    import fire
    fire.Fire()
