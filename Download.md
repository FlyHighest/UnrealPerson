# Download Synthesized Data 

Our synthesized data (named Unreal in the paper) is generated with Makehuman, Mixamo, and UnrealEngine 4. We provide 1.2M images of 4.8K identities, captured from 4 unreal environments. 

Beihang Netdisk: [Download Link](https://bhpan.buaa.edu.cn:443/link/BD6502DF5A2A2434BC5FC62793F80F96) valid until: 2024-01-01

BaiduPan: [Download Link](https://pan.baidu.com/s/1P_UKdhmuDvJNQHuO81ifww) password: abcd

Google Drive: [Download Link](https://drive.google.com/drive/folders/1sQHVBWvDwn-SVJtMqZDpk2HK9g9tnZ1I?usp=sharing)

The image path is formulated as: unreal_v{X}.{Y}/images/{P}\_c{D}_{F}.jpg,
 for example, unreal_v3.1/images/333_c001_78.jpg.
 
_X_ represents the ID of unreal environment; _Y_ is the version of human models; _P_ is the person identity label; _D_ is the camera label; _F_ is the frame number. 

We provide three types of human models: version 1 is the basic type; version 2 contains accessories, like handbags, hats and backpacks; version 3 contains hard samples with similar global appearance. 
Four virtual environments are used in our synthesized data: the first three are city environments and the last one is a supermarket.
Note that cameras under different virtual environments may have the same label and persons of different versions may also have the same identity label. 
Therefore, images with the same (Y, P) belong the the same virtual person; images with the same (X, D) belong the same camera. 
 
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

