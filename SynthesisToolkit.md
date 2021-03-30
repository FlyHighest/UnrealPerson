# Data Synthesis Toolkit

The data synthesis toolkit includs three parts: Makehuman plugin, several UE4 blueprints and data annotation scripts.
In this tutorial, we show how to generate more data that meets your own needs with our toolkit. 


## 1. Prepare 3D Human Models 

**The following steps are optional if our prepared human models are sufficient for your demand. **

3D human models download link: [Beihang Pan](??).

Steps to generate your own customized models: 

(1) Download or clone [makehuman v1.2](https://github.com/makehumancommunity/makehuman/releases/tag/v1.2.0). 

(2) Put the directory "9_massproduce" under the directory of makehuman/plugins. The massproduce is a [community plugin](https://github.com/makehumancommunity/community-plugins-massproduce) of Makehuman, used to generate random 3D human models by batch. We modify it to support more configs about clothing types. 

(3) Put clothing assets under the directory of makehuman data dir: `PathToUserDocumentDir/makehuman(win/linux) or MakeHuman(macOS)/v1py3/data/clothes`. You can download the assets using the script `script_makehuman_asset_download.py` and annotate the asset folder like this:
```
[GenderSymbol]_[ClothingType]_[The original asset folder name]
```
`GenderSymbol` can be `m`,`f`,or `mf`, indicating this asset is suitable for male only, female only or both male & female. 
`ClothingType` can be one of `up`,`low`,`full`,`fullo`,`shoe`,`mask`,`scarf`,`glasses`,`handheld`,`headwear`,`backpack`,`blank`. Specially, you need to prepare several assets that are invisible when applied on 3D models. They are used to occupy the material slots. The aligned material slots make it easy for us to change materials later in UE4.   

Optionally, you can also download our annotated assets from this [link](???). 

(4) After all three steps above, you can now (a) restart your makehuman program, (b) go to tab "Geometries->Eyes" and select "Low-poly" (otherwise it looks weird in UE4), (c) go to tab "Pose/Animate->Skeleton" and select "Game engine", (d) go to tab "Community->Mass produce", in "Export settings" choose file format "FBX", in the middle panel and the left panel choose your preferred configurations, click "Produce" after entering the number of models you want. 


## 2. Prepare UE4 and UnrealCV plugin. 



(1) Install UE4. version ...

(2) Compile UnrealCV. we modify the plugin ...

(3) Prepare some virtual scenes ... 

## 3. Prepare animations, textures, virtual scenes

(1) Textures, download or cut. 

(2) animations, download or go to mixamo.

(3) Purchase or make your own

## 4. Data Synthesis

(1) Character BP: MHModel\Divide

(2) Level BP: Director

(3) Start game

(4) Start script

## 5. Post process

(1) Cut to images

(2) for video-based ReID: combine to tracklets






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