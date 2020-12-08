from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pickle
import os
import sys
import random
import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import opt
from io_stream import data_manager

from frameworks.models import ResNetBuilder
from frameworks.evaluating import evaluator_manager

from utils.serialization import Logger, load_previous_model, load_moco_model
from utils.transforms import TestTransform
import os

def test(**kwargs):
    opt._parse(kwargs)
    if opt.save_dir.startswith('pytorch'):
        opt.save_dir=os.path.split(opt.save_dir)[1]
    save_file = 'log_test_{}_{}.txt'.format(opt.testset_name,opt.testepoch)
    if opt.testset_name == 'unreal_test':
        save_file = 'log_test_{}_{}_{}.txt'.format(opt.testset_name,opt.testepoch,opt.datasets)

    sys.stdout = Logger(
        os.path.join("./pytorch-ckpt/current", opt.save_dir, save_file))
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    print('initializing dataset {}'.format(opt.testset_name))
    if opt.testset_name != 'unreal_test':
        dataset = data_manager.init_dataset(name=opt.testset_name,
                                        num_bn_sample=opt.batch_num_bn_estimatation * opt.test_batch)

    if opt.testset_name=='unreal_test':
        dataset = data_manager.init_unreal_dataset(name=opt.testset_name,
                                                         datasets = opt.datasets, 
                                                         num_pids=opt.num_pids,
                                                         img_per_person = opt.img_per_person,
                                        num_bn_sample=opt.batch_num_bn_estimatation * opt.test_batch)
    pin_memory = True if use_gpu else False

    print('loading model from {} ...'.format(opt.save_dir))
    model = ResNetBuilder()
    if opt.model_path is not None:
        model_path = opt.model_path
    else:    
        model_path = os.path.join("./pytorch-ckpt/current", opt.save_dir,
                              '{}.pth.tar'.format(opt.testepoch))
    if opt.model_path is not None and opt.model_path.find('moco')!=-1:
        model = load_moco_model(model,model_path)
    else:
        model = load_previous_model(model, model_path, load_fc_layers=False)
    model.eval()

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    reid_evaluator = evaluator_manager.init_evaluator(opt.testset_name, model, flip=True)

    def _calculate_bn_and_features(all_data, sampled_data):
        time.sleep(1)
        all_features, all_ids, all_cams,all_paths = [], [], [], []
        available_cams = list(sampled_data)
        cam_bn_info = dict()
        for current_cam in tqdm.tqdm(available_cams):
            camera_data = all_data[current_cam]
            if len(camera_data)==0:
                continue
            camera_samples = sampled_data[current_cam]
            data_for_camera_loader = DataLoader(
                data_manager.init_datafolder(opt.testset_name, camera_samples, TestTransform(opt.height, opt.width),with_path=True),
                batch_size=opt.test_batch, num_workers=opt.workers,
                pin_memory=False, drop_last=True
            )
            bn_info = reid_evaluator.collect_sim_bn_info(data_for_camera_loader)
            cam_bn_info[current_cam]=bn_info
            camera_data = all_data[current_cam]
            if len(camera_data)==0:
                continue
            data_loader = DataLoader(
                data_manager.init_datafolder(opt.testset_name, camera_data, TestTransform(opt.height, opt.width), with_path =True),
                batch_size=opt.test_batch, num_workers=opt.workers,
                pin_memory=pin_memory, shuffle=False
            )
            fs, pids, camids, img_paths = reid_evaluator.produce_features(data_loader, normalize=True)
            all_features.append(fs)
            all_ids.append(pids)
            all_cams.append(camids)
            all_paths.extend(img_paths)

        all_features = torch.cat(all_features, 0)
        all_ids = np.concatenate(all_ids, axis=0)
        all_cams = np.concatenate(all_cams, axis=0)
        
        time.sleep(1)

        pickle.dump(cam_bn_info,open('cam_bn_info-{}-{}.pkl'.format(opt.save_dir,opt.testset_name),'wb'))
        return all_features, all_ids, all_cams, all_paths

    print('Processing query features...')
    qf, q_pids, q_camids, q_paths = _calculate_bn_and_features(dataset.query_per_cam, dataset.query_per_cam_sampled)
    print('Processing gallery features...')
    gf, g_pids, g_camids, g_paths = _calculate_bn_and_features(dataset.gallery_per_cam,
                                                      dataset.gallery_per_cam_sampled)
    if opt.testset_name =='msmt_sepcam':
        cid2label = dataset.cid2label
        label2cid = dict()
        for c,l in cid2label.items():
            label2cid[l]=c[0]
        print(label2cid)
        q_cids = list()
        for qc in q_camids:
            q_cids.append(label2cid[qc])
        g_cids = list()
        for gc in g_camids:
            g_cids.append(label2cid[gc])
        q_camids = np.asarray(q_cids)
        g_camids = np.asarray(g_cids)

    pickle.dump({'qp':q_paths,'gp':g_paths},open('paths.pkl','wb'))
    print('Computing CMC and mAP...')
    reid_evaluator.get_final_results_with_features(qf, q_pids, q_camids, gf, g_pids, g_camids, q_paths,g_paths)


if __name__ == '__main__':
    import fire

    fire.Fire()
