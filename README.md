# Face Brightness

This repo is mainly sharing the code of face-skin-brightness metric and brightness-information-metric in the [Face Recognition Accuracy Across Demographics: Shining a Light Into the Problem](https://arxiv.org/pdf/2206.01881.pdf). Also it includes the code for calculating the imposter and genuine distributions, face feature extraction, and plots.

## Face-skin-brightness metric

This metric is used to measure the brightness level of an face image in terms of the upper face skin, which provides more accurate brightness measurement than the [commercial SDK](https://www.innovatrics.com/iface-sdk/).

Boundaries: [94.92973987, 115.86682531, 198.77210088, 220.54641581]
<p align="center" width="100%">
    <img width="55%" src="https://github.com/SteveXWu/Face_Brightness/blob/main/images/fsb.png"> 
    <img width="43%" src="https://github.com/SteveXWu/Face_Brightness/blob/main/images/iFace_fsb_comparison.png"> 
</p>


#### Implementation

Make sure the [face-parsing](https://github.com/SteveXWu/face-parsing.PyTorch) package is downloaded. Note that this package should be placed at the same directory level as your project, otherwise the path in [degree_separation.py](https://github.com/SteveXWu/Face_Brightness/blob/main/degree_separation.py) should be changed.

```markdown
.
|---face-parsing
|---project
```

Run brightness analyzing code

```shell
python depree_separation.py
```

## Brightness-information-metric

This metric is used to measure the brightness variance of the face skin in order to reflect the information on the face.
$$BIM=\sum_{i=0}^{N}|B_i-\bar{B}|P(B_i)$$

#### Implementation

```shell
python brightness_information.py
```

## Imposter and genuine distributions

We analyzed the effect of over-and-under exposed on the performance of the state-of-the-art model [ArcFace](https://github.com/deepinsight/insightface) (paper link: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf)). The distributions of African-American male are as below:

<p align="center" width="100%">
    <img width="80%" src="https://github.com/SteveXWu/Face_Brightness/blob/main/images/Arc_AA_M_mix.png"> 
</p>

More detailed analyses can be found in [Face Recognition Accuracy Across Demographics: Shining a Light Into the Problem](https://arxiv.org/pdf/2206.01881.pdf).

## Citation

If you find any of the tools useful in your research, please consider to cite this paper:

```
@inproceedings{wu2023face,
  title={Face recognition accuracy across demographics: Shining a light into the problem},
  author={Wu, Haiyu and Albiero, V{\'\i}tor and Krishnapriya, KS and King, Michael C and Bowyer, Kevin W},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1041--1050},
  year={2023}
}
```

