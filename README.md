# EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone

[**EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone**](https://arxiv.org/abs/2307.05463)    
[Shraman Pramanick](https://shramanpramanick.github.io/), [Yale Song](http://people.csail.mit.edu/yalesong/home/), [Sayan Nag](https://sayannag.github.io/), [Kevin Qinghong Lin](https://qinghonglin.github.io/), [Hardik Shah](https://www.linkedin.com/in/hardik-shah-75ab5429/), [Mike Z. Shou](https://sites.google.com/view/showlab), [Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/), [Pengchuan Zhang](https://pzzhang.github.io/pzzhang/)                
ICCV, 2023               
[arxiv](https://arxiv.org/pdf/2307.05463.pdf) | [project page](https://shramanpramanick.github.io/EgoVLPv2/)

> **TL;DR:** We introduce the second generation of egocentric video-language pre-training (EgoVLPv2), a significant improvement from the previous generation, by incorporating cross-modal fusion directly into the video and language backbones.

<img src="/Figures/EgoVLPv2_System.gif" alt="EgoVLPv2" style="zoom:67%;" />

## 📢 News

- [September, 2023] We release the EgoVLPv2 codebase, checkpoints and features.
- [July, 2023] EgoVLPv2 is accepted in **ICCV 2023**.

## 📁 Repository Structure

The contents of this repository are structured as follows:

```bash
EgoVLPv2
    ├── EgoVLPv2
    │   ├── Pre-training on EgoClip version of Ego4D
    │   ├── Validation on EgoMCQ 
    │   ├── Zero-Shot and fine-tuning on EK-100 MIR
    │   ├── Zero-shot and fine-tuning on Charades-Ego
    │   └── Feature extraction on EgoMQ
    ├── EgoTaskQA
    │   └── Fine-tuning on EgoTaskQA direct and indirect splits
    ├── EgoNLQ
    │   └── Feature extraction and head-tuning on EgoNLQ 
    ├── QFVS
    │   └── Feature extraction and head-tuning on QFVS
    └── EgoMQ
        └── Head-tuning on EgoMQ 
```

Each directory contains data settings, training/inference scripts, and checkpoints. Notably, we provided pre-extracted video and text features to power Ego4D NLQ & MQ challenges.

## 🛠️ Environment Preparation

```bash
conda create -n python=3.8.13 egovlpv2 pip
conda activate egovlpv2
pip install -r requirements.txt
```

## ✉️ Contact
This repository is created and maintained by [Shraman](https://shramanpramanick.github.io/). Questions and discussions are welcome via spraman3@jhu.edu.
We are willing to merge results if EgoVLPv2 is transferred to other egocentric tasks or datasets.

## 🙏 Acknowledgements
The codebase for this work is built on the [EgoVLP](https://github.com/showlab/EgoVLP/tree/f3e8895c7a1a691bc7fb0c07618c3be0015887eb), [LAVILA](https://github.com/facebookresearch/LaViLa), [FIBER](https://github.com/microsoft/FIBER), and [VSLNet](https://github.com/26hzhang/VSLNet) repository. We would like to thank the respective authors for their contribution, and the Meta AI team for discussions and feedback.

## 📄 License

EgoVLPv2 is licensed under a [MIT License](./LICENSE).

## 🎓 Citing EgoVLPv2

```
@article{pramanick2023egovlpv2,
  title={EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone},
  author={Pramanick, Shraman and Song, Yale and Nag, Sayan and Lin, Kevin Qinghong and Shah, Hardik and Shou, Mike Zheng and Chellappa, Rama and Zhang, Pengchuan},
  journal={arXiv preprint arXiv:2307.05463},
  year={2023}
}
```
