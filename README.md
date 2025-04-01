# SIAVC
SIAVC: Semi-Supervised Framework for Industrial Accident Video Classification

我们提供了一个用于工业事故视频分类的半监督算法。

PS. SIAVC's 2D version is coming soon

<p align="center">
  <img src="Overview.png" width="750px"/>
</p>

# Dataset
The ECA9 dataset is released at [here]( https://pan.baidu.com/s/1acHbQiuaPE1DZtBsyxibzQ?pwd=vrea). 




<p align="center">
  <img src="ECA9.png" width="750px"/>
</p>

You can update your own data as:
```
/dataset 
  /dataset's name 
    /train
      /class1
        11.mp4 or 11.mp4.pkl
        ...
      /class2
        21.mp4 or 21.mp4.pkl
        ...
      ...
    /test
      /class1
        111.mp4 or 111.mp4.pkl
        ...
      /class2
        211.mp4 or 211.mp4.pkl
        ...
      ...

```    

# Train

SIAVC.py contains code for training and testing.

SIAVC.py 中已经包含了训练以及测试的代码，设置好数据集路径之后运行即可。

```
run SIAVC.py
```

# Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2405.14506).
```
@misc{li2024siavc,
      title={SIAVC: Semi-Supervised Framework for Industrial Accident Video Classification}, 
      author={Zuoyong Li and Qinghua Lin and Haoyi Fan and Tiesong Zhao and David Zhang},
      year={2024},
      eprint={2405.14506},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
