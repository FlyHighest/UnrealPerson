import glob
import re
import numpy as np
import random
from os import path as osp
from collections import defaultdict
from io_stream.data_utils import reorganize_images_by_camera


class SyRI(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'syri'

    def __init__(self, root='data', **kwargs):
        self.name = 'syri'
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = self.dataset_dir
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = [],0,0 #self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = [],0 ,0 # self._process_dir(self.gallery_dir, relabel=False)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> SyRI loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        #self.query_per_cam, self.query_per_cam_sampled = reorganize_images_by_camera(self.query,
        #                                                                             kwargs['num_bn_sample'])
        #self.gallery_per_cam, self.gallery_per_cam_sampled = reorganize_images_by_camera(self.gallery,
        #                                                                                 kwargs['num_bn_sample'])

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_([\d]+)_')
        dataset = []
        pid_container=set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            camid = camid//2   # index starts from 0
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
