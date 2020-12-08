import numpy as np
import os, math


def joint_similarity(qf,qc,qfr,gf,gc,gfr,distribution):
    query = qf
    score = np.dot(gf,query)
    gamma = 5

    interval = 100
    score_st = np.zeros(len(gc))
    for i in range(len(gc)):
        if qfr>gfr[i]:
            diff = qfr-gfr[i]
            hist_ = int(diff/interval)
            pr = distribution[qc-1][gc[i]-1][hist_]
        else:
            diff = gfr[i]-qfr
            hist_ = int(diff/interval)
            pr = distribution[gc[i]-1][qc-1][hist_]
        score_st[i] = pr

    score  = 1-1/(1+np.exp(-gamma*score))*1/(1+2*np.exp(-gamma*score_st))
    return score

def compute_joint_dist(distribution, q_feas, g_feas, q_frames, g_frames, q_cams, g_cams):
    dists = []
    for i in range(len(q_frames)):
        dist = joint_similarity(
            q_feas[i],q_cams[i],q_frames[i], 
            g_feas,   g_cams,   g_frames,
            distribution)
        
        dist = np.expand_dims(dist, axis=0)
        dists.append(dist)
        #print(i, dist.shape)
    dists = np.concatenate(dists, axis=0)

    return dists

    


def gaussian_func(x, u, o=50):
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(o, 2))
    return temp1 * np.exp(temp2)

def gauss_smooth(arr,o):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i

    approximate_delta = 3*o     #  when x-u>approximate_delta, e.g., 6*o, the gaussian value is approximately equal to 0.
    gaussian_vect= gaussian_func(vect,0,o)
    matrix = np.zeros((hist_num,hist_num))
    for i in range(hist_num):
        k=0
        for j in range(i,hist_num):
            if k>approximate_delta:
                continue
            matrix[i][j]=gaussian_vect[j-i] 
            k=k+1  
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2
            
    xxx = np.dot(matrix,arr)
    return xxx

def get_st_distribution(camera_id, labels, frames, id_num, cam_num=8):
    spatial_temporal_sum = np.zeros((id_num,cam_num))                       
    spatial_temporal_count = np.zeros((id_num,cam_num))
    eps = 0.0000001
    interval = 100.0    
    
    for i in range(len(camera_id)):
        label_k = int(labels[i])              #### not in order, done
        cam_k = int(camera_id[i]-1)           ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]

        spatial_temporal_sum[label_k][cam_k]=spatial_temporal_sum[label_k][cam_k]+frame_k
        spatial_temporal_count[label_k][cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    spatial_temporal_avg = spatial_temporal_sum/(spatial_temporal_count+eps)  # spatial_temporal_avg: 702 ids, 8cameras, center point
    
    distribution = np.zeros((cam_num,cam_num,3000))
    for i in range(id_num):
        for j in range(cam_num-1):
            for k in range(j+1,cam_num):
                if spatial_temporal_count[i][j]==0 or spatial_temporal_count[i][k]==0:
                    continue                   
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij>st_ik:
                    diff = st_ij-st_ik
                    hist_ = int(diff/interval)
                    distribution[j][k][hist_] = distribution[j][k][hist_]+1     # [big][small]
                else:
                    diff = st_ik-st_ij
                    hist_ = int(diff/interval)
                    distribution[k][j][hist_] = distribution[k][j][hist_]+1

    for i in range(id_num):
        for j in range(cam_num):
            if spatial_temporal_count[i][j] >1:
                
                frames_same_cam = []
                for k in range(len(camera_id)):
                    if labels[k]==i and camera_id[k]-1 ==j:
                        frames_same_cam.append(frames[k])
                frame_id_min = min(frames_same_cam)

                #print 'id, cam, len',i, j, len(frames_same_cam)
                for item in frames_same_cam:
                    #if item != frame_id_min:                    
                    diff = item - frame_id_min
                    hist_ = int(diff/interval)
                    #print item, frame_id_min, diff, hist_
                    distribution[j][j][hist_] = distribution[j][j][hist_] + spatial_temporal_count[i][j]                       
    
    smooth = 50
    for i in range(cam_num):
        for j in range(cam_num):
            #print("gauss "+str(i)+"->"+str(j))
            distribution[i][j][:]=gauss_smooth(distribution[i][j][:],smooth)

    sum_ = np.sum(distribution,axis=2)
    for i in range(cam_num):
        for j in range(cam_num):
            distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps) 
    
    return distribution    # [to][from], to xxx camera, from xxx camera



