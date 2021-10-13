import glob,shutil
import tqdm
import os
import sys,re,traceback
import numpy as np
from io import BytesIO
import time
import threading
import random
import PIL.Image
import pickle
import glob
from multiprocessing import Process

class HandleImageThread(Process):
    def __init__(self, dir_save, dir_data, dir_data_mask, cam, ptype_name):
        super(HandleImageThread, self).__init__()
        self.dir_save=dir_save
        self.dir_data=dir_data
        self.dir_data_mask = dir_data_mask
        self.cam=cam
        self.ptype_name = ptype_name


    def run(self):
        datautils = DataUtils(self.dir_save, self.dir_data, self.dir_data_mask, self.ptype_name)
        datautils.get_object_color()


        files = glob.glob(os.path.join(self.dir_save ,"c{:0>3d}_*lit.png".format(self.cam)))
        files.sort()

        for fi in tqdm.tqdm(files):
            datautils.generate_one_frame(fi)

# TODO: replace this with a better implementation
class Color(object):
    ''' A utility class to parse color value '''
    def __init__(self, color_str):
        self.color_str = color_str
        self.R,self.G,self.B,self.A=0,0,0,0
        color_str = color_str.replace("(","").replace(")","").replace("R=","").replace("G=","").replace("B=","").replace("A=","")
        try:
            (self.R, self.G, self.B, self.A) = [int(i) for i in color_str.split(",")]
        except Exception as e:
            print("Error in Color:")
            print(color_str)

    def __repr__(self):
        return self.color_str

class DataUtils(object):
    def __init__(self,dir_save,dir_data,dir_data_mask, ptype_name):
        self.ptype_name = ptype_name
        self.dir_save = dir_save
        tmp = os.path.basename(os.path.split(self.dir_save)[0])
        self.dir_data = os.path.join(dir_data, tmp)
        self.dir_data_mask = os.path.join(dir_data_mask, tmp)
        try:
            if not os.path.exists(self.dir_data):
                os.makedirs(self.dir_data)
            if not os.path.exists(self.dir_data_mask):
                os.makedirs(self.dir_data_mask)
        except Exception as execeptione:
            pass
        self.id2color = {}


    def read_png(self,res):
        import PIL.Image
        img = PIL.Image.open(res)
        return np.asarray(img)

    def read_pngIMG(self,res):
        import PIL.Image
        img = PIL.Image.open(res)
        return img

    def get_object_color(self):
        self.id2color = pickle.load(open(self.dir_save+"object_color.pkl",'rb'))

    def match_color(self,object_mask, target_color, tolerance=3):
        match_region = np.ones(object_mask.shape[0:2], dtype=bool)
        for c in range(3): # r,g,b
            min_val = target_color[c] - tolerance
            max_val = target_color[c] + tolerance
            channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
            match_region &= channel_region

        if match_region.sum() > 2000:
            return match_region
        else:

            return None

    def generate_one_frame(self,lit_file):

        lit = self.read_pngIMG(lit_file)

        object_mask = self.read_png(lit_file.replace("lit","mask"))

        s=os.path.split(lit_file)[1]
        cam = s.split("_")[0]
        frame =s.split("_")[1]

        id2mask = {}
        for obj_id in self.id2color.keys():
            color = self.id2color[obj_id]
            mask = self.match_color(object_mask, [color.R, color.G, color.B], tolerance=3)
            if mask is not None:
                id2mask[obj_id] = mask
        # This may take a while
        # TODO: Need to find a faster implementation for this
        count = 0
        for k, x in id2mask.items():
            left = 9999
            top = 9999
            right = 0
            bottom = 0
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if x[i][j] == True:
                        if i > bottom:
                            bottom = i
                        if i < top:
                            top = i
                        if j < left:
                            left = j
                        if j > right:
                            right = j
            if left < 5 or top <5 or bottom >= x.shape[0] - 5 or right >= x.shape[1] - 5:
                continue

            r1 = right-left
            r2 = bottom - top
            left = max(0, left-random.randint(0,int(0.1*r1)))
            right= min(x.shape[1]-1, right+random.randint(0,int(0.1*r1)))
            top = max(0, top-random.randint(0,int(0.1*r2)))
            bottom= min(x.shape[0]-1, bottom+random.randint(0,int(0.1*r2)))

            img = lit.crop((left, top, right, bottom))

            img = img.convert('RGB')

            # mask save disabled
            x_part = x[top:bottom, left:right]
            img_mask = np.zeros((bottom-top,right-left,3),dtype=np.uint8)
            img_mask[x_part==True]=[255,255,255]
            img_mask = PIL.Image.fromarray(img_mask)

            qqq = np.sum(x)/(r1*r2)
            if (1.0*(right-left))/(1.0*(bottom-top)) > 0.7:
                continue

            if (right-left) * (bottom-top) < 3000:
                continue

            if np.sum(x)/(r1*r2) < 0.3:
                continue
            try:
                img.save(os.path.join(self.dir_data,k.replace(self.ptype_name,"") + "_{}_F{}_{}_{}_{}_{}.jpg"
                         .format(cam,frame, left,top,right,bottom)))
                # mask save
                img_mask.save(os.path.join(self.dir_data_mask,k.replace(self.ptype_name,"")+ "_{}_F{}_{}_{}_{}_{}.png"
                         .format(cam,frame, left,top,right,bottom)))
                count += 1
            except AttributeError or SystemError as e:
                continue





if __name__=="__main__":
    try:
        import os,tqdm,glob
        import argparse
        parser= argparse.ArgumentParser()
        parser.add_argument("--path",type=str)
        parser.add_argument("--scene",type=str,choices=['s001','s002','s003','s004'])
        parser.add_argument("--ptype",type=int)
        cam_num = {'s001': 6,
                   's002': 16,
                   's003': 6,
                   's004': 6}
        ptype_name = {1: "MHMasterAI_C_",
                      2: "MHMainClass_C_",
                      3: "MHMainClassDivide_C_"}
        args = parser.parse_args()

        save_dir = 'unreal_video_{}_p{}'.format(args.scene, args.ptype)

        dir_saves = [i + '\\' for i in glob.glob(os.path.join(args.path, "tmp*"))]

        dir_data = "F:\\video\\{}\\images\\".format(save_dir)
        dir_data_mask = "F:\\video\\{}\\annos\\".format(save_dir)
        if not os.path.exists(dir_data):
            os.makedirs(dir_data)
        if not os.path.exists(dir_data_mask):
            os.makedirs(dir_data_mask)
        l =[]
        for dir_save in dir_saves:
            for c in range(1, 1+cam_num[args.scene]):
                thread = HandleImageThread(dir_save+"\\",dir_data,dir_data_mask,c, ptype_name[args.ptype])
                thread.start()
                l.append(thread)

        for t in l:
            t.join()


    except Exception as e:

        traceback.print_exc()