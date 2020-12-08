import os, torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from scipy.spatial.distance import cdist

from utils.util import cluster, get_info
from utils.util import extract_fea_camtrans, extract_fea_test, extract_fea_test_cbn
from utils.resnet import resnet50
from utils.dataset import imgdataset, imgdataset_camtrans
from utils.rerank import re_ranking
from utils.st_distribution import get_st_distribution
from utils.evaluate_joint_sim import evaluate_joint
import argparse
import sys
from utils.logger import Logger

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--weights',type=str)
parser.add_argument('--name',type=str)
parser.add_argument('--tar',default='market',type=str, help='target domain')
parser.add_argument('--num_cam', default=8, type=int, help='target camera number')
parser.add_argument('--joint', default=True, type=bool, help='joint similarity or visual similarity')

opt = parser.parse_args()
sys.stdout=Logger(os.path.join('./snapshot',opt.name,'log_test_{}.txt'.format(os.path.split(opt.weights)[1])))

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

dataset_path = 'data/'
ann_file_train = 'list_{}/list_{}_train.txt'.format(opt.tar,opt.tar)
ann_file_test = 'list_{}/list_{}_test.txt'.format(opt.tar,opt.tar)

snapshot = opt.weights
#'snapshot/resnet50_{}2{}_epoch{}_cbn.pth'.format(opt.src,opt.tar,opt.epoch)

num_cam = opt.num_cam
###########   DATASET   ###########
img_dir = dataset_path + '{}/bounding_box_train_camstyle_merge/'.format(opt.tar)
train_dataset = imgdataset_camtrans(dataset_dir=img_dir, txt_path=ann_file_train, 
	transformer='test', K=num_cam, num_cam=num_cam)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=4)

img_dir = dataset_path + '{}/'.format(opt.tar)
test_dataset = imgdataset(dataset_dir=img_dir, txt_path=ann_file_test, transformer='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

###########   TEST   ###########
model, _ = resnet50(pretrained=snapshot, num_classes=2)
model.cuda()
model.eval()

print('extract feature for testing set')
test_feas,pids,cids,fids,is_q = extract_fea_test_cbn(model, img_dir, ann_file_test) 

if opt.joint:
    print('extract feature for training set')
    train_feas, _,cam_ids, frames , _ = extract_fea_test_cbn(model,dataset_path + '{}/bounding_box_train_camstyle_merge/'.format(opt.tar),ann_file_train)
    print('generate spatial-temporal distribution')
    dist = cdist(train_feas, train_feas)
    dist = np.power(dist,2)
    #dist = re_ranking(original_dist=dist)
    labels = cluster(dist)
    num_ids = len(set(labels))
    print('cluster id num:', num_ids)
    distribution = get_st_distribution(cam_ids, labels, frames, id_num=num_ids, cam_num=num_cam)
else:
    distribution = None

print('evaluation')
evaluate_joint(test_fea=test_feas, st_distribute=distribution, ann_file=(pids,cids,fids,is_q))
