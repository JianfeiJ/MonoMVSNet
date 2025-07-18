# **MonoMVSNet**

Code comming soon.

## [Arxiv](https://arxiv.org/abs/2507.11333)

> MonoMVSNet: Monocular Priors Guided Multi-View Stereo Network  
> Authors: Jianfei Jiang, Qiankun Liu*, Haochen Yu, Hongyuan Liu, Liyong Wang, Jiansheng Chen, Huimin Ma*   
> Institute: University of Science and Technology Beijing  
> ICCV 2025  

## Abstract
Learning-based Multi-View Stereo (MVS) methods aim to predict depth maps for a sequence of calibrated images to recover dense point clouds. However, existing MVS methods often struggle with challenging regions, such as textureless regions and reflective surfaces, where feature matching fails. In contrast, monocular depth estimation inherently does not require feature matching, allowing it to achieve robust relative depth estimation in these regions. To bridge this gap, we propose MonoMVSNet, a novel monocular feature and depth guided MVS network that integrates powerful priors from a monocular foundation model into multi-view geometry. Firstly, the monocular feature of the reference view is integrated into source view features by the attention mechanism with a newly designed cross-view position encoding. Then, the monocular depth of the reference view is aligned to dynamically update the depth candidates for edge regions during the sampling procedure. Finally, a relative consistency loss is further designed based on the monocular depth to supervise the depth prediction. Extensive experiments demonstrate that MonoMVSNet achieves state-of-the-art performance on the DTU and Tanks-and-Temples datasets, ranking first on the Tanks-and-Temples Intermediate and Advanced benchmarks.

<p align="center">
<img src="assets/overview.png" width="100%">
</p>

## **Results**

### **Quantitative Results on DTU**


| DTU | Acc. ↓ | Comp. ↓ | Overall ↓ |
|:---:|:------:|:-------:|:---------:|
| Ours (N=5) | 0.313 | 0.243 | 0.278 |
| Ours (N=9) | 0.302 | 0.248 | 0.275 |

---
### **Quantitative Results on Tanks-and-Temples**

| T&T (Inter.) | Mean ↑ | Family | Francis | Horse | Lighthouse | M60 | Panther | Playground | Train |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ours | 68.63 | 82.38 | 72.89 | 62.80 | 70.49 | 65.79 | 68.54 | 65.54 | 60.59 |


| T&T (Adv.) | Mean ↑ | Auditorium | Ballroom | Courtroom | Museum | Palace | Temple |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ours | 43.58 | 30.33 | 46.76 | 42.90 | 56.31 | 37.28 | 47.88 |


## Citation
If you find this work useful in your research, please consider citing the following preprint:
```bibtex

```
