# SIAVC
SIAVC: Semi-Supervised Framework for Industrial Accident Video Classification

<p align="center">
  <img src="Overview.png" width="750px"/>
</p>

# Dataset
The ECA9 dataset will be released soon.

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
