# UnrealPerson: An Adaptive Pipeline for Costless Person Re-identification 
In our paper ([pdf](https://arxiv.org/abs/2012.04268)), we propose a novel pipeline, UnrealPerson, that decreases the costs in both the training and deployment stages of person ReID. 
We develop an automatic data synthesis toolkit and use synthesized data in mutiple ReID tasks, including (i) Direct transfer, (ii) Unsupervised domain adaptation, and (iii) Supervised fine-tuning. 


**Highlights:**
1. In direct transfer evaluation, we achieve 38.5% rank-1 accuracy on MSMT17 and 79.0% on Market-1501 using our unreal data. 
2. In unsupervised domain adaptation, we achieve 68.2% rank-1 accuracy on MSMT17 and 93.0% on Market-1501 using our unreal data. 
3. We obtain a better pre-trained ReID model with our unreal data.  

**This repo contains:**
1. The **synthesized data** we use in the paper, including more than 6,000 identities in 4 different virtual scenes. Please see this document [Download.md](Download.md) for details.
2. The codes for our experiments reported in the paper. To reproduce the results on direct transfer, supervised learning and unsupervised domain adaptation, please refer to this document: [Experiments.md](Experiments.md) .
3. The necessary scripts and detailed tutorials to help you generate your own data! [SynthesisToolkit.md](SynthesisToolkit.md)


## Demonstration

![](imgs/unrealperson.jpg)


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
