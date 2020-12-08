import shutil
import glob
import os
from tqdm import tqdm 

train_files = open('list_train.txt','r')

txt = open('list_msmt_train.txt','w')

temporal_slot = ['0113morning','0113noon','0113afternoon',
                         '0114morning','0114noon','0114afternoon',
                         '0302morning','0302noon','0302afternoon',
                         '0303morning','0303noon','0303afternoon']

for f in tqdm(train_files.readlines()):
    if 'fake' in f or len(f.split('_'))<5:
        continue
    else:
        
        pid = f.split('/')[0]
        cam = f.split('_')[2]
        img_num = f.split("_")[1]
        slot = f.split("_")[3]

        frame = f.split("_")[4]

        frame = int(frame)+10000*temporal_slot.index(slot)


        new_name = '{}_c{}_0{}.jpg'.format(pid,int(cam),img_num)
        txt.write('{} {} {} {}\n'.format(new_name, pid, cam, frame))

txt.close()


