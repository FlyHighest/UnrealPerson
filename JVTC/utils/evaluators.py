import torch, cv2, os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .ranking import cmc, mean_ap

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def extract_features(model, data_loader, select_set='market'):
	if select_set == 'market':
		query_num = 3368
	elif select_set =='duke':
		query_num = 2228

	print(select_set, "feature extraction start")
	model.eval()
	features = []
	labels = []
	cams = []
	for i, (images, label, cam)  in enumerate(data_loader):
		with torch.no_grad():
			out = model(Variable(images).cuda())
			features.append(out[1])
			labels.append(label)
			cams.append(cam)

	features = torch.cat(features).cpu().numpy()
	labels = torch.cat(labels).cpu().numpy()
	cams = torch.cat(cams).cpu().numpy()
	#print('features', features.shape, labels.shape, cams.shape)

	query_labels = labels[0:query_num]
	query_cams = cams[0:query_num]

	gallery_labels = labels[query_num:]
	gallery_cams = cams[query_num:]

	query_features = features[0:query_num, :]
	gallery_features = features[query_num:, :]
	print("extraction done, feature shape:", np.shape(features))
	
	return (query_features, query_labels, query_cams), (gallery_features, gallery_labels, gallery_cams)


def evaluate_all(model, data_loader, select_set='market'):
	query, gallery = extract_features(model, data_loader, select_set=select_set)

	query_features, query_labels, query_cams = query
	gallery_features, gallery_labels, gallery_cams = gallery

	dist = np.zeros((query_features.shape[0], gallery_features.shape[0]), dtype = np.float64)
	for i in range(query_features.shape[0]):
		dist[i, :] = np.sum((gallery_features-query_features[i,:])**2, axis=1)

	mAP = mean_ap(dist, query_labels, gallery_labels, query_cams, gallery_cams)
	cmc_scores = cmc(dist, query_labels, gallery_labels, query_cams, gallery_cams,
		separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
	print('mAP: %.4f, r1:%.4f, r5:%.4f, r10:%.4f, r20:%.4f'%(mAP, cmc_scores[0], cmc_scores[4], cmc_scores[9], cmc_scores[19]))

	return None
