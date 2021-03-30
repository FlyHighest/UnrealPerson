# Experiments of UnrealPerson

The codes are based on [CBN](https://github.com/automan000/Camera-based-Person-ReID) (ECCV 2020) and [JVTC](https://github.com/ljn114514/JVTC) (ECCV 2020). 

### Direct Transfer and Supervised Fine-tuning

We use Camera-based Batch Normalization baseline for direct transfer and supervised fine-tuning experiments.
  
**1. Clone this repo and change directory to CBN**
```bash
git clone https://github.com/FlyHighest/UnrealPerson.git
cd UnrealPerson/CBN
```

**2. Download Market-1501, DukeMTMC-reID, MSMT17, UnrealPerson data and organize them as follows:**
<pre>
.
+-- data
|   +-- market
|       +-- bounding_box_train
|       +-- query
|       +-- bounding_box_test
|   +-- duke
|       +-- bounding_box_train
|       +-- query
|       +-- bounding_box_test
|   +-- msmt17
|       +-- train
|       +-- test
|       +-- list_train.txt
|       +-- list_val.txt
|       +-- list_query.txt
|       +-- list_gallery.txt
|   +-- unreal_vX.Y
|       +-- images
|   +-- unreal_vX.Y
|       +-- images
+ -- other files in this repo
</pre>



**3. Install the required packages**
```console
pip install -r requirements.txt
```


**4. Put the official PyTorch [ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) pretrained model to your home folder: 
'~/.torch/models/'**


**5. Train a ReID model with our synthesized data**

Reproduce the results in our paper:

```console
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python train_model.py train --trainset_name unreal --datasets='unreal_v1.1,unreal_v2.1,unreal_v3.1,unreal_v4.1,unreal_v1.2,unreal_v2.2,unreal_v3.2,unreal_v4.2,unreal_v1.3,unreal_v2.3,unreal_v3.3,unreal_v4.3' --save_dir='unreal_4678_v1v2v3_cambal_3000' --save_step 15  --num_pids 3000 --cam_bal True --img_per_person 40
```

We also provide the trained weights of this experiment in the data download links above.

Configs:
When ``trainset_name`` is unreal, ``datasets`` contains the directories of unreal data that will be used. ``num_pids`` is the number of humans and ``cam_bal`` denotes the camera balanced sampling strategy is adopted. ``img_per_person`` controls the size of the training set.

More configurations are in [config.py](https://github.com/FlyHighest/UnrealPerson/CBN/config.py).

**6.1 Direct transfer to real datasets**
```console
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python test_model.py test --testset_name market --save_dir='unreal_4678_v1v2v3_cambal_3000'
```

**6.2 Fine-tuning**
```console
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0 \
python train_model.py train --trainset_name market --save_dir='market_unrealpretrain_demo' --max_epoch 60 --decay_epoch 40 --model_path pytorch-ckpt/current/unreal_4678_v1v2v3_cambal_3000/model_best.pth.tar


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python test_model.py test --testset_name market --save_dir='market_unrealpretrain_demo'
```


### Unsupervised Domain Adaptation

We use joint visual and temporal consistency (JVTC) framework. CBN is also implemented in JVTC.

**1. Clone this repo and change directory to JVTC**

```bash
git clone https://github.com/FlyHighest/UnrealPerson.git
cd UnrealPerson/JVTC
```

**2. Prepare data**

Basicly, it is the same as CBN, except for an extra directory ``bounding_box_train_camstyle_merge``, which can be downloaded from [ECN](https://github.com/zhunzhong07/ECN). We suggest using ``ln -s`` to save disk space. 
<pre>
.
+-- data
|   +-- market
|       +-- bounding_box_train
|       +-- query
|       +-- bounding_box_test
|       +-- bounding_box_train_camstyle_merge
+ -- other files in this repo
</pre>

**3. Install the required packages**

```console
pip install -r ../CBN/requirements.txt
```


**4. Put the official PyTorch [ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) pretrained model to your home folder: 
'~/.torch/models/'**

**5. Train and test**

(Unreal to MSMT)

```console
python train_cbn.py --gpu_ids 0,1,2 --src unreal --tar msmt --num_cam 6 --name unreal2msmt --max_ep 60

python test_cbn.py --gpu_ids 1 --weights snapshot/unreal2msmt/resnet50_unreal2market_epoch60_cbn.pth --name 'unreal2msmt' --tar market --num_cam 6 --joint True 
```

The unreal data used in JVTC is defined in list_unreal/list_unreal_train.txt. The CBN codes support generating this file (see CBN/io_stream/datasets/unreal.py). 

More details can be seen in [JVTC](https://github.com/ljn114514/JVTC).

### References

- [1] Rethinking the Distribution Gap of Person Re-identification with Camera-Based Batch Normalization. ECCV 2020.

- [2] Joint Visual and Temporal Consistency for Unsupervised Domain Adaptive Person Re-Identification. ECCV 2020.


## Cite our paper

If you find our work useful in your research, please kindly cite:

```
@inproceedings{unrealperson,
      title={UnrealPerson: An Adaptive Pipeline towards Costless Person Re-identification}, 
      author={Tianyu Zhang and Lingxi Xie and Longhui Wei and Zijie Zhuang and Yongfei Zhang and Bo Li and Qi Tian},
      year={2021},
      booktitle={CVPR}
}
```

If you have any questions about the data or paper, please leave an issue or contact me:
zhangtianyu@buaa.edu.cn