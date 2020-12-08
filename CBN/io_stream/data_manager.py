from __future__ import print_function, absolute_import

from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from io_stream.datasets.market import Market1501
from io_stream.datasets.msmt import MSMT17
from io_stream.datasets.duke import Duke
from io_stream.datasets.randperson import RandPerson
from io_stream.datasets.combine import Combine
from io_stream.datasets.unreal import Unreal
from io_stream.datasets.syri import SyRI
from io_stream.datasets.personx import PersonX

class ReID_Data(Dataset):
    def __init__(self, dataset, transform, with_path=False):
        self.dataset = dataset
        self.transform = transform
        self.with_path = with_path

    def __getitem__(self, item):
        img_path, pid, camid = self.dataset[item]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.with_path:
            return img,pid,camid,img_path
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)



"""Create datasets"""

__data_factory = {
    'market': Market1501,
    'duke': Duke,
    'msmt': MSMT17,
    'randperson':RandPerson,
    'combine':Combine,
    'unreal':Unreal,
    'syri':SyRI,
    'personx':PersonX
}

__folder_factory = {
    'market': ReID_Data,
    'duke': ReID_Data,
    'msmt': ReID_Data,
}

def init_unreal_dataset(name,datasets,*args,**kwargs):
    sets = datasets.split(',') if type(datasets)!=tuple else datasets
    dataset=[]
    for s in sets:
        dataset.append(s)
    
    return __data_factory[name](dataset=dataset,*args,**kwargs)

def init_combine_dataset(name,options,datasets,*args,**kwargs):
    sets=datasets.split(',') if type(datasets)!=tuple else datasets
    dataset=[]
    for s in sets:
        if s=='unreal':
            dataset.append(init_unreal_dataset(s,datasets=options.dataset,num_pids=options.num_pids,
                                                         img_per_person = options.img_per_person))
        else:
            dataset.append(init_dataset(s,*args,**kwargs))
    
    return Combine(dataset=dataset,*args,**kwargs)

def init_dataset(name, *args, **kwargs):
    if name not in __data_factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __data_factory[name](*args, **kwargs)


def init_datafolder(name, data_list, transforms,with_path=False):
    if name not in __folder_factory.keys():
        return ReID_Data(data_list,transforms,with_path)
    return __folder_factory[name](data_list, transforms,with_path)
