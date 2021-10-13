import glob, os, shutil
from PIL import Image
from tqdm import tqdm
input_image_dir = "F:\\video\\unreal_video_bunker_psameclo75"
full_image_dir = "F:\\datasets"
save_path = "F:\\unreal_data\\unreal_video_bunker_psameclo75"

def get_image_info(path):
    basename = os.path.basename(path)
    arrs = basename.split('_')
    pid, cid, frame, left, top, right, bottom = arrs[0], arrs[1], int(arrs[2][1:]), arrs[3],arrs[4],arrs[5],arrs[6][:-4]
    full_image_path = os.path.join(full_image_dir,path.split("\\")[-2])+"\\{}_{}_lit.png".format(cid,frame)
    return pid, cid, frame, (int(left), int(top), int(right), int(bottom)), full_image_path

def save_tracklet(tid, img_list):
    save_dir = os.path.join(save_path, str(tid))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    normal_count = 0
    extra_count = 0
    for i in range(len(img_list)):
        img, frame, bbox, full_img_path = img_list[i]
        shutil.copy(img, os.path.join(save_dir, os.path.basename(img)))
        normal_count+=1
        if i == len(img_list)-1:
            break
        else:
            next_frame = img_list[i+1][1]
            next_bbox = img_list[i+1][2]
            if next_frame-frame>1:
                diff = next_frame-frame

                bbox_extras = [(bbox[0] + (next_bbox[0] - bbox[0]) / (next_frame - frame) * x,
                              bbox[1] + (next_bbox[1] - bbox[1]) / (next_frame - frame) * x,
                              bbox[2] + (next_bbox[2] - bbox[2]) / (next_frame - frame) * x,
                              bbox[3] + (next_bbox[3] - bbox[3]) / (next_frame - frame) * x)
                             for x in range(1,diff)]
                full_img_extras = [full_img_path.replace("{}_lit".format(frame),"{}_lit".format(frame+x))
                                  for x in range(1, diff)]

                for j in range(len(full_img_extras)):
                    full_img_extra=full_img_extras[j]
                    full_img = Image.open(full_img_extra)
                    img_extra = full_img.crop(bbox_extras[j]).convert('RGB')
                    img_extra.save(os.path.join(save_dir, os.path.basename(img.replace("F{}".format(frame),"F{}_extra".format(frame+j+1)))))
                    extra_count+=1

    return normal_count,extra_count


images = glob.glob(os.path.join(input_image_dir, 'images\\*\\*.jpg'))

images.sort()

current_pid, current_cid, current_frame = None, None, None
current_image_list = []
tid = 0
pid_container = set()
normal = 0
extra = 0
for img in tqdm(images):
    pid, cid, frame, bbox, full_img_path = get_image_info(img)

    pid_container.add(pid)
    if pid==current_pid and cid==current_cid and frame-current_frame<10:
        current_image_list.append((img,frame, bbox, full_img_path))
        current_frame=frame
    else:
        # save current list
        if len(current_image_list)>0:
            nc,ec = save_tracklet(tid, current_image_list)
            normal+=nc
            extra+=ec
            tid += 1
        current_image_list =  []
        current_pid=pid
        current_cid=cid
        current_frame=frame
        current_image_list.append((img,frame, bbox, full_img_path))

print("Tracklet: ",tid)
print("Pids: ",len(pid_container))
print("Normal: ",normal )
print("Extra: ",extra)


