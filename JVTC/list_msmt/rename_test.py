import shutil
import glob
import os
from tqdm import tqdm 

test_files = open('list_test.txt','r')

txt = open('list_msmt_test.txt','w')

temporal_slot = ['0113morning','0113noon','0113afternoon',
                         '0114morning','0114noon','0114afternoon',
                         '0302morning','0302noon','0302afternoon',
                         '0303morning','0303noon','0303afternoon']
# query/0000/0000_000_01_0303morning_0015_0.jpg 0 01 0015
for f in tqdm(test_files.readlines()):
    if True:    
        pid = f.split('/')[1]
        cam = f.split('_')[2]
        slot = f.split("_")[3]

        frame = f.split("_")[4]

        frame = int(frame)+10000*temporal_slot.index(slot)


        new_name = f.split(' ')[0]
        txt.write('{} {} {} {}\n'.format(new_name, pid, cam, frame))

txt.close()


