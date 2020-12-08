from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import warnings


class DefaultConfig(object):
    seed = 0
    share_cam = False
    num_pids=800
    num_cams=0
    img_per_person = 60
    model_path =None
    # dataset options
    trainset_name = 'market'
    testset_name = 'duke'
    height = 256
    width = 128
    # sampler
    workers = 8
    num_instances = 4
    # default optimization params
    train_batch = 64
    test_batch = 32
    max_epoch = 15
    decay_epoch = 10
    save_step = max_epoch
    # estimate bn statistics
    batch_num_bn_estimatation = 10
    # io
    datasets='market,unreal_v6+v4+v7'
    dataset = 'unreal_v4.1,unreal_v6.1,unreal_v7.1,unreal_v8.1'
    print_freq = 50
    loss ='softmax'
    save_dir = './pytorch-ckpt/market'
    cam_bal=False
    testepoch='model_best'
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
