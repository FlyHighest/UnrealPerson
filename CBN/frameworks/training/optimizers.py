from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def get_our_optimizer_strategy(opt, optim_policy=None):
    base_lr = 1e-2
    if opt.loss=='triplet':
        base_lr=2e-4
    optimizer = torch.optim.SGD(
        optim_policy, lr=base_lr, weight_decay=5e-4, momentum=0.9
    )
    if opt.loss=='softmax':
        def adjust_lr(optimizer, ep):
            if ep < opt.decay_epoch:
                lr = 1e-2
            else:
                lr = 1e-3
            for i, p in enumerate(optimizer.param_groups):
                p['lr'] = lr
            return lr
    elif opt.loss=='triplet':
        def adjust_lr(optimizer, ep):
            lr=1e-3
            if ep >=opt.decay_epoch:       
                lr = 1e-3 * (0.001 ** (float(ep + 1 - opt.decay_epoch)/ (opt.max_epoch + 1 - opt.decay_epoch)))
            for p in optimizer.param_groups:
                p['lr'] = lr
            return lr


    return optimizer, adjust_lr
