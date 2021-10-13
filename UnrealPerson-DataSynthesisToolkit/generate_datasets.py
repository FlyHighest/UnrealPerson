from unrealcv import client
import sys,re,traceback
import numpy as np
from io import BytesIO
import time
import pickle,random
from tqdm import trange
import argparse
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
    def __init__(self,client,dir_save):
        self.client = client
        self.client.connect()
        self.dir_save = dir_save
        if not client.isconnected():
            print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
            sys.exit(-1)
        res = self.client.request('vget /unrealcv/status')
        print(res)
        self.scene_objects=None
        self.id2color = {}

    def read_png(self,res):
        import PIL.Image
        img = PIL.Image.open(BytesIO(res))
        return np.asarray(img)

    def read_pngIMG(self,res):
        import PIL.Image
        img = PIL.Image.open(BytesIO(res))
        return img

    def get_object_color(self):
        self.scene_objects = self.client.request('vget /objects').split(' ')
        print('Number of objects in this scene:', len(self.scene_objects))
        self.id2color = {}  # Map from object id to the labeling color
        for obj_id in self.scene_objects:
            if obj_id not in self.id2color.keys():
                if obj_id.startswith("MH"):
                    color = Color(self.client.request('vget /object/{}/color'.format(obj_id)))
                    self.id2color[obj_id] = color
        pickle.dump(self.id2color,open(self.dir_save+"object_color.pkl",'wb'))

    def match_color(self,object_mask, target_color, tolerance=3):
        match_region = np.ones(object_mask.shape[0:2], dtype=bool)
        for c in range(3): # r,g,b
            min_val = target_color[c] - tolerance
            max_val = target_color[c] + tolerance
            channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
            match_region &= channel_region

        if match_region.sum() != 0:
            return match_region
        else:
            return None

    def generate_one_frame(self,cam,frame):
        self.client.request('vrun ce Pause')
        res = self.client.request('vget /camera/0/lit png')
        lit = self.read_pngIMG(res)
        #print('The image is saved to {}'.format(res))
        res = self.client.request('vget /camera/0/object_mask png')
        object_mask = self.read_pngIMG(res)
        self.client.request('vrun ce Resume')
        lit.save(self.dir_save  + "{}_{}_lit.png".format(cam, frame))
        object_mask.save(self.dir_save  + "{}_{}_mask.png".format(cam, frame))

        # print('%s : %s' % (obj_id, str(color)))
        #
        # id2mask = {}
        # for obj_id in self.scene_objects:
        #     if obj_id.startswith("MH"):
        #         color = self.id2color[obj_id]
        #         mask = self.match_color(object_mask, [color.R, color.G, color.B], tolerance=3)
        #         if mask is not None:
        #             id2mask[obj_id] = mask
        # # This may take a while
        # # TODO: Need to find a faster implementation for this
        #
        # for k, x in id2mask.items():
        #     if k.startswith("MH"):
        #         left = 9999
        #         top = 9999
        #         right = 0
        #         bottom = 0
        #         for i in range(x.shape[0]):
        #             for j in range(x.shape[1]):
        #                 if x[i][j] == True:
        #                     if i > bottom:
        #                         bottom = i
        #                     if i < top:
        #                         top = i
        #                     if j < left:
        #                         left = j
        #                     if j > right:
        #                         right = j
        #         img = lit.crop((left, top, right, bottom))
        #         if (right-left) * (bottom-top)<1500:
        #             continue
        #         try:
        #             img.save(self.dir_save+k + "_{}_{}.png".format(cam,frame))
        #         except AttributeError or SystemError as e:
        #             continue





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",type=str,choices=['s001','s002','s003','s004'])
    parser.add_argument("--person",type=int)
    parser.add_argument("--images",type=int) #images per camera

    opt = parser.parse_args()

    person_per_batch = {'s001': 100,
                        's002': 100,
                        's003': 100,
                        's004': 50}
    light_condition =  {'s001': 1,
                        's002': 12,
                        's003': 6,
                        's004': 1}
    try:
        import os,glob
        datautils = DataUtils(client, "")
        cam_info = glob.glob('caminfo_'+opt.scene+"*.pkl")
        cam_lr = []
        for cam_info_file in cam_info:
            cam_lr.extend(pickle.load(open(cam_info_file, 'rb')))

        cam = ["c{:0>3d}".format(x) for x in range(1, 1+len(cam_lr))]
        print("Found {} cameras".format(len(cam_lr)))
        batch = opt.person // person_per_batch[opt.scene] + 1
        print(cam_lr)
        for _ in range(batch):
            datautils.dir_save = "f:/video/tmp"+str(int(time.time()))+"/"
            os.mkdir(datautils.dir_save)
            datautils.client.request("vrun ce DelActor")
            datautils.client.request("vrun ce AddActor")
            datautils.get_object_color()
            time.sleep(60)
            for cc in range(len(cam_lr)):
                if light_condition[opt.scene] > 1:
                    datautils.client.request("vrun ce LCreset")

                datautils.client.request('vset /camera/0/location '+cam_lr[cc][0].strip())
                datautils.client.request('vset /camera/0/rotation ' + cam_lr[cc][1].strip())
                for i in trange(opt.images):
                    if i % opt.images//light_condition[opt.scene] == 0 and light_condition[opt.scene]>1:
                        datautils.client.request("vrun ce LC")
                    datautils.generate_one_frame(cam[cc], i)

    except Exception as e:
        datautils.client.disconnect()
        traceback.print_exc()
    finally:
        datautils.client.disconnect()



