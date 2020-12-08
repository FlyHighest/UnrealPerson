from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import matplotlib
import pickle
matplotlib.use('Agg')
from tqdm import tqdm
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool
from collections import defaultdict

def mt_eval_func(query_id, query_cam, gallery_ids, gallery_cams, order, matches, max_rank):

    remove = (gallery_ids[order] == query_id) & (gallery_cams[order] == query_cam)
    keep = np.invert(remove)
    orig_cmc = matches[keep]
    if not np.any(orig_cmc):
        return -1, -1
    
    cmc = orig_cmc.cumsum()
    cmc[cmc > 1] = 1
    single_cmc = cmc[:max_rank]

    num_rel = orig_cmc.sum()
    tmp_cmc = orig_cmc.cumsum()
    tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
    tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
    single_ap = tmp_cmc.sum() / num_rel
    return single_ap, single_cmc, orig_cmc[:max_rank]


class BaseEvaluator(object):
    def __init__(self, model):
        self.model = model
        self.eval_func = self.eval_func1

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs):
        raise NotImplementedError

    def evaluate(self, queryloader, galleryloader, ranks):
        raise NotImplementedError


    def eval_func1(self, distmat, q_pids, g_pids, q_camids, g_camids, q_paths,g_paths,max_rank=50):
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        mt_pool = Pool(cpu_count())
        results = []
        path_dict = dict()
        for q_idx in range(num_q):
            path_dict[q_idx] = indices[q_idx][:30]
            params = (q_pids[q_idx], q_camids[q_idx], g_pids, g_camids, indices[q_idx], matches[q_idx], max_rank)
            results.append(mt_pool.apply_async(mt_eval_func, params))

        mt_pool.close()
        mt_pool.join()
        res=[]
        for x in results:
            if len(x.get())==3:
                res.append(x.get())
        results = res #[x.get() for x in results]
        
        all_AP = np.array([x[0] for x in results])
        valid_index = all_AP > -1
        all_AP = all_AP[valid_index]
        all_cmc = np.array([x[1] for x in results])
        all_cmc = all_cmc[valid_index, ...]
        num_valid_q = len(all_AP)
        try:
            all_ranks = np.array([x[2] for x in results])
        except Exception as e:
            from IPython import embed
            embed()
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        pickle.dump(path_dict,open('path_dict.pkl','wb'))
        print(num_valid_q)
        return all_cmc, mAP, all_ranks
