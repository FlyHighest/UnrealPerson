from io import BytesIO

import numpy as np
import sys,pickle

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




        # The image resolution and port is configured in the config file.
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
