from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torch.nn import functional as F
from itertools import chain

from torch.nn import DataParallel
from torch.nn.parallel.scatter_gather import scatter, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from .dataset import imgdataset, imgdataset_cam

class CamDataParallel(DataParallel):
    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        all_inputs = inputs[0]
        all_kwargs = kwargs
        all_outputs = []

        while len(all_inputs) > 0:
            num_required_gpu = min(len(all_inputs), len(self.device_ids))
            actual_inputs = [all_inputs.pop(0) for _ in range(num_required_gpu)]
            inputs, kwargs = self.scatter(actual_inputs, all_kwargs, self.device_ids[:num_required_gpu])
            replicas = self.replicate(self.module, self.device_ids[:num_required_gpu])
            all_outputs.extend(self.parallel_apply(replicas, inputs, kwargs))

        return self.gather(all_outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, input_list, kwargs, device_ids):
        inputs = []
        for input, gpu in zip(input_list, device_ids):
            inputs.extend(scatter(input, [gpu], dim=0))
        kwargs = scatter(kwargs, device_ids, dim=0) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def organize_data(input_data,camids,input_labels):
    unique_camids = torch.unique(camids).cpu().numpy()
    reorg_data = []
    reorg_labels = []
    for cid in unique_camids:
        current_camid = (camids == cid).nonzero().view(-1)
        if current_camid.size(0) > 1:
            data = torch.index_select(input_data, index=current_camid, dim=0)
            labels = torch.index_select(input_labels, index=current_camid, dim=0)
            reorg_data.append(data)
            reorg_labels.append(labels)
    return reorg_data,torch.cat(reorg_labels,dim=0)

def l2_dist(fea_query, fea_gallery):
    dist = np.zeros((fea_query.shape[0], fea_gallery.shape[0]), dtype = np.float64)
    for i in range(fea_query.shape[0]):
        dist[i, :] = np.sum((fea_gallery-fea_query[i,:])**2, axis=1)
    return dist
    

def cluster(dist, rho=1.6e-3):

    tri_mat = np.triu(dist, 1)
    tri_mat = tri_mat[np.nonzero(tri_mat)]
    tri_mat = np.sort(tri_mat,axis=None)
    top_num = np.round(rho*tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()
    #low eps for training without source domain
    #eps = eps*0.9
    #print('eps in cluster: {:.3f}'.format(eps))
    cluster = DBSCAN(eps=eps,min_samples=1,metric='precomputed', n_jobs=8)
    labels = cluster.fit_predict(dist)

    return labels


def get_info(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        #self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in lines]
        labels = [int(i.split()[1]) for i in lines]
        cam_ids = [int(i.split()[2]) for i in lines]
        frames = [int(i.split()[3]) for i in lines]

    return labels, cam_ids, frames

def extract_fea_camtrans(model, loader):
    feas = []
    for i, data in enumerate(loader, 1):
        #break
        with torch.no_grad():
            image = data[0].cuda()

            batch_size = image.size(0)
            K = image.size(1)

            image = image.view(image.size(0)*image.size(1), image.size(2), image.size(3), image.size(4))
            #image = Variable(image).cuda()
            out = model(image)
            fea = out[2]
            fea = fea.view(batch_size, K, -1)
            fea = fea.mean(dim=1)
            fea = F.normalize(fea)
            feas.append(fea)

    feas = torch.cat(feas)
    #print('duke_train_feas', feas.size())
    return feas.cpu().numpy()

def extract_fea_test_cbn(model,img_dir, ann_file_path):
    # divide dataset according to cam id
    test_dataset = imgdataset(dataset_dir=img_dir, txt_path=ann_file_path, transformer='test')
    cams = test_dataset.cams
    fea, pid, cid, fid, is_q = [],[],[],[],[]
    for c in cams:
        test_dataset_cam = imgdataset_cam(dataset_dir=img_dir,txt_path = ann_file_path, camid=c, transformer='test')
        test_loader = DataLoader(dataset=test_dataset_cam, batch_size=64, shuffle=True, num_workers=4)        
        collect_sim_bn_info(model,test_loader)
        test_loader = DataLoader(dataset=test_dataset_cam, batch_size=64, shuffle=False, num_workers=0)        
        for i ,data in enumerate(test_loader):
            inputs, label, cam, frame, is_query = data
            inputs = inputs.cuda()
            feature = None
            for i in range(2):
                if i == 1:
                    inputs = flip_tensor_lr(inputs)
                with torch.no_grad():
                    global_f = model(inputs)[1]

                if feature is None:
                    feature = global_f
                else:
                    feature += global_f
            feature=feature/2
            fea.append(feature)
            pid.extend(label)
            cid.extend(cam)
            fid.extend(frame)
            is_q.extend(is_query)
    fea = torch.cat(fea,0).cpu().numpy()
    pid = np.asarray(pid)
    cid = np.asarray(cid)
    fid = np.asarray(fid)
    is_q= np.asarray(is_q)
    return fea,pid,cid,fid,is_q


def extract_fea_test(model, loader):
    feas = []
    for i, data in enumerate(loader, 1):
        #break
        with torch.no_grad():
            image = data[0].cuda()
            out = model(image)
            fea = out[1]
            feas.append(fea)

    feas = torch.cat(feas)
    #print('duke_train_feas', feas.size())
    return feas.cpu().numpy()

def flip_tensor_lr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def collect_sim_bn_info(model, dataloader, num=10):
    network_bns = [x for x in list(model.modules()) if
                       isinstance(x, torch.nn.BatchNorm2d) or isinstance(x, torch.nn.BatchNorm1d)]
    for bn in network_bns:
        bn.running_mean = torch.zeros(bn.running_mean.size()).float().cuda()
        bn.running_var = torch.ones(bn.running_var.size()).float().cuda()
        bn.num_batches_tracked = torch.tensor(0).cuda().long()

    model.train()
    for batch_idx, inputs in enumerate(dataloader):
        if batch_idx==num:
            break
        # each camera should has at least 2 images for estimating BN statistics
        assert len(inputs[0].size()) == 4 and inputs[0].size(
            0) > 1, 'Cannot estimate BN statistics. Each camera should have at least 2 images'
        image = inputs[0].cuda()
        for i in range(2):
            if i == 1:
                image = flip_tensor_lr(image)
            with torch.no_grad():
                model(image)
    model.eval()

