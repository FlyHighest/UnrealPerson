import torch
import torch.nn as nn

def SetLr(lr, optimizer):
	for g in optimizer.param_groups:
		g['lr'] = lr * g.get('lr_mult', 1)	

def StepLrUpdater(epoch, base_lr=0.01, gamma=0.1, step=[8,11]):
	if isinstance(step, int):
		return base_lr * (gamma**(epoch // step))

	exp = len(step)
	for i, s in enumerate(step):
		if epoch < s:
			exp = i
			break
	return base_lr * gamma**exp


def warmup_lr(cur_iters, warmup_iters=500, warmup_type='linear', warmup_ratio=1/3):

	if warmup_type == 'constant':
		warmup_lr = warmup_ratio
	elif warmup_type == 'linear':
		warmup_lr = 1 - (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
	elif warmup_type == 'exp':
		warmup_lr = warmup_ratio**(1 - cur_iters / warmup_iters)
	return warmup_lr

