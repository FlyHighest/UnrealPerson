import os, torch, random, cv2, math, glob
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
from torch.nn import functional as F

from collections import defaultdict
import random
import copy
from torch.utils.data.sampler import Sampler


class IdentityCameraSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances,cams_of_dataset=None,len_of_real_data=None):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances  # approximate
        self.num_cams_per_batch = 8
        self.index_dic = defaultdict(list)
        self.cam_index_dic = dict()
        self.num_pids_per_cam = self.num_pids_per_batch//self.num_cams_per_batch
        for index, (_, pid, camid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            if camid not in self.cam_index_dic.keys():
                self.cam_index_dic[camid]=defaultdict(list)
            self.cam_index_dic[camid][pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.cams_of_dataset=cams_of_dataset
        self.len_of_real_data = len_of_real_data

    def __iter__(self):
        final_idxs = []
        length = 2*self.len_of_real_data if self.len_of_real_data is not None else len(self.data_source)
        # F setting
        #length = len(self.data_source)
        while(len(final_idxs) < length):
            if self.cams_of_dataset is not None:
                # C setting
                #c_rnd = np.random.choice(list(self.cam_index_dic.keys()),size=1)[0]
                #for cams_of_data in self.cams_of_dataset:
                #    if c_rnd in cams_of_data:
                #        cams = np.random.choice(list(cams_of_data),size=self.num_cams_per_batch,replace=True)
                #        break

                # D setting
                c_rnd = np.random.choice([i for i in range(len(self.cams_of_dataset))],size=1)[0]
                cams = np.random.choice(list(self.cams_of_dataset[c_rnd]),size=self.num_cams_per_batch,replace=True)

            else:
                cams = np.random.choice(list(self.cam_index_dic.keys()),size=self.num_cams_per_batch,replace=True)
            for c in cams:
                pids = np.random.choice(list(self.cam_index_dic[c].keys()),size=self.num_pids_per_cam, replace=True)
                for p in pids:
                    idxs =np.random.choice(self.cam_index_dic[c][p],size=self.num_instances,replace=True)
                    random.shuffle(idxs)
                    final_idxs.extend(idxs)
        self.length=len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                return img

        return img

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = T.Compose([
    T.Resize((256,128)),
    T.RandomHorizontalFlip(),
    T.ToTensor(), 
    normalizer,
    RandomErasing(EPSILON=0.5) 
    ])

test_transform = T.Compose([
    T.Resize((256,128)),
    T.ToTensor(),  
    normalizer  ])

class imgdataset_withsource(data.Dataset):
    def __init__(self, data_source):
        self.data_source = data_source
        self.transform = train_transform
    
    def __getitem__(self,index):
        im_path, pid, cam = self.data_source[index]
        image = Image.open(im_path).convert('RGB')
        image = self.transform(image)
        return image,pid, cam

    def __len__(self):
        return len(self.data_source)

class imgdataset(data.Dataset):
    def __init__(self, dataset_dir, txt_path, transformer = 'train'):
        self.mode = transformer
        self.transform = train_transform if transformer == 'train' else test_transform
        with open(txt_path) as f:
            line = f.readlines()
            self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in line]
            self.label_list = [int(i.split()[1]) for i in line]
            self.cam_list = [int(i.split()[2]) for i in line]
            if self.mode=='test':
                self.frame_list = [int(i.split()[3]) for i in line]
            #self.cam_list = [int(i.split('c')[1][0]) for i in line]
        self.cams = np.unique(self.cam_list)
        self.pids = np.unique(self.label_list)
        
        pid2label = {pid:ind for ind,pid in enumerate(self.pids)}
        labels = []
        for l in self.label_list:
            labels.append(pid2label[l])
        self.label_list = labels

        self.data_source = []
        for i in range(len(self.label_list)):
            self.data_source.append((self.img_list[i],self.label_list[i],self.cam_list[i]))
      

    def __getitem__(self, index):
        im_path = self.img_list[index]
        image = Image.open(im_path).convert('RGB')             
        image = self.transform(image)             
        if self.mode=='train':
            return image, self.label_list[index], self.cam_list[index]
        elif self.mode=='test':
            return image, self.label_list[index], self.cam_list[index], self.frame_list[index]

    def __len__(self):
        return len(self.label_list)


class imgdataset_cam(data.Dataset):
    def __init__(self, dataset_dir, txt_path,camid, transformer = 'train'):
        self.mode = transformer
        self.transform = train_transform if transformer == 'train' else test_transform
        with open(txt_path) as f:
            line = f.readlines()
            self.img_list = np.array([os.path.join(dataset_dir, i.split()[0]) for i in line])
            self.label_list = np.array([int(i.split()[1]) for i in line])
            self.cam_list = np.array([int(i.split()[2]) for i in line])
            self.query_list = np.array([True if 'query' in i else False for i in line])
            if self.mode=='test':
                self.frame_list =np.array([int(i.split()[3]) for i in line])
            select = self.cam_list==camid
            self.img_list = self.img_list[select]
            self.label_list = self.label_list[select]
            self.cam_list = self.cam_list[select]
            self.frame_list = self.frame_list[select]
            self.query_list = self.query_list[select]
            #self.cam_list = [int(i.split('c')[1][0]) for i in line]
        self.cams = np.unique(self.cam_list)

    def __getitem__(self, index):
        im_path = self.img_list[index]
        image = Image.open(im_path).convert('RGB')             
        image = self.transform(image)             
        if self.mode=='train':
            return image, self.label_list[index], self.cam_list[index]
        elif self.mode=='test':
            return image, self.label_list[index], self.cam_list[index], self.frame_list[index], self.query_list[index]

    def __len__(self):
        return len(self.label_list)







class imgdataset_camtrans(data.Dataset):
    def __init__(self, dataset_dir, txt_path, transformer = 'train', num_cam=8, K=4):
        self.num_cam = num_cam
        self.mode = transformer
        self.transform = train_transform if transformer == 'train' else test_transform
        self.K = K
        with open(txt_path) as f:
            line = f.readlines()
            self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in line]
            self.label_list = [int(i.split()[1]) for i in line]
            #self.cam_list = [int(i.split('c')[1][0]) for i in line]
            self.cam_list = [int(i.split()[2]) for i in line]

    def __getitem__(self, index):
        im_path = self.img_list[index]
        camid = self.cam_list[index]
        cams = torch.randperm(self.num_cam) + 1

        imgs = []
        cam_labels = []
        index_labels = []
        for sel_cam in cams[0:self.K]:
            
            if sel_cam != camid:
                if 'msmt' in im_path:
                    im_path_cam = im_path[:-4]+'_fake_'+str(sel_cam.numpy())+'.jpg'

                else:
                    im_path_cam = im_path[:-4] + '_fake_' + str(camid) + 'to' + str(sel_cam.numpy()) + '.jpg'
            else:
                im_path_cam = im_path
            
            #print('im_path', camid, sel_cam,im_path_cam)
            image = Image.open(im_path_cam).convert('RGB')
            image = self.transform(image)
            imgs.append(image.numpy())
            #imgs.append(image)
            cam_labels.append(sel_cam)
            index_labels.append(index)

        imgs = np.array(imgs, np.float32)
        imgs = torch.from_numpy(imgs).float()
        cam_labels = np.array(cam_labels)
        cam_labels = torch.from_numpy(cam_labels)
        index_labels = np.array(index_labels)
        index_labels = torch.from_numpy(index_labels)
        return imgs, self.label_list[index], index_labels, cam_labels

    def __len__(self):
        return len(self.label_list)


class NormalCollateFn:
    def __call__(self, batch):
        img_tensor = [x[0] for x in batch]
        pids = np.array([x[1] for x in batch])
        camids = np.array([x[2] for x in batch])
        return torch.stack(img_tensor, dim=0), torch.from_numpy(pids), torch.from_numpy(np.array(camids))

