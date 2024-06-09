# SIAVC
SIAVC: Semi-Supervised Framework for Industrial Accident Video Classification

我们提供了一个用于工业事故视频分类的半监督算法。

<p align="center">
  <img src="Overview.png" width="750px"/>
</p>

# Dataset
The ECA9 dataset will be released soon.

我们构建了一个涵盖18类风险行为、9类典型安全事故的工业生成安全监控数据集(8000+小时)。但遗憾的是，由于签署了保密协议，我仅提供论文中所使用数据的pkl文件。

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
