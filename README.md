<div align="center">

# 🥟 SAMOSA

### Segment Anything with Motion, Geometry, and Semantic Adaptation for Complex Nonlinear Visual Object Tracking

[Deyi Zhu](https://github.com/DurYi/)<sup>&ast;</sup>,
[Yuji Wang](https://voyagewang.github.io/)<sup>&ast;</sup>,
[Yong Liu](https://yongliu20.github.io/),
[Yansong Tang](https://andytang15.github.io/),
[Bingyao Yu](https://yuby14.github.io/),
[Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/),
[Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en)

Tsinghua University

<sub><sup>&ast;</sup> Equal contribution</sub>

<p align="center">
  <a href="https://arxiv.org/abs/2605.22538"><img src="https://img.shields.io/badge/arXiv-2605.22538-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/facebookresearch/sam2"><img src="https://img.shields.io/badge/Built%20on-SAM%202-blue.svg" alt="SAM 2"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776ab.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.3.1-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/status-under%20review-orange.svg" alt="Status">
</p>

**Official repository** for the [paper](https://arxiv.org/abs/2605.22538)
_"Segment Anything with Motion, Geometry, and Semantic Adaptation for Complex Nonlinear Visual Object Tracking."_

</div>

---

## 📖 Overview

Traditional visual object tracking (VOT) methods typically rely on task-specific supervised training, which limits their generalization to unseen objects and to challenging scenarios involving distractors, occlusion, and nonlinear motion. Recent vision foundation models — exemplified by **SAM 2** — learn strong video-understanding priors from large-scale pretraining and offer a promising foundation for building more robust and generalizable trackers. However, directly applying SAM 2 to VOT remains suboptimal: it does not explicitly model target motion dynamics, nor does it enforce geometric and semantic consistency across frames, both of which are essential for reliable tracking.

To address this, we propose **SAMOSA**, a tracking framework that adapts SAM 2 to complex VOT scenarios by explicitly leveraging **motion**, **geometry**, and **semantic** cues:

- A **lightweight nonlinear Motion Predictor** models target dynamics and guides both mask selection and memory filtering.
- **Semantic cues** detect target shifts and enable recovery from tracking failures.
- **Geometric cues** act as structural constraints to improve tracking stability.

In this way, SAMOSA bridges the gap between SAM 2's implicit video-understanding prior and explicit tracking-oriented modeling. Extensive experiments show that SAMOSA consistently outperforms state-of-the-art SAM 2–based approaches on general benchmarks, demonstrates stronger generalization than supervised VOT methods, and achieves substantial gains on anti-UAV datasets, which typify complex nonlinear motion scenarios.

## ✨ Highlights

- **🎯 Higher-order Markov Motion Predictor (MP).** Models nonlinear target motion and, together with an **Error Detection–Recovery Module (EDRM)**, explicitly identifies potential tracking failures and mitigates error propagation.
- **🧠 Target-Aware Memory Bank (TAMB).** Adaptively selects representative and reliable memory frames, guided by confidence, occlusion, and motion cues.
- **🏆 State-of-the-art performance.** Strong results across general VOT benchmarks (LaSOT<sub>ext</sub>, OTB, TrackingNet) and challenging anti-UAV tracking benchmarks, with notable improvements in nonlinear-motion scenarios.
- **⚡ Lightweight and easy to integrate.** MP is the **only trainable component**; it is trained solely on annotated bounding-box trajectories — without video frames — and plugs into SAM 2 at inference time with limited latency overhead.

## 🗺️ Roadmap

- [x] **Done** — Our paper is available on [arXiv](https://arxiv.org/abs/2605.22538)!
- [x] **Done** — Test scripts for more benchmarks released.
- [x] **Done** — Raw results released on [Google Drive](https://drive.google.com/file/d/1loIjhCcQcjVlgbPvryaDefDE4a1tdEUS/view?usp=sharing).
- [ ] **Incoming** — Release training code for the Motion Predictor.
- [ ] **Incoming** — Release a demo script to support inference on video.

## 🚀 Getting Started

### 1. SAMOSA Installation

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install both the PyTorch and TorchVision dependencies. You can install **the SAMOSA version** of SAM 2 on a GPU machine using:

```bash
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

> 💡 Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.

Install the other requirements:

```bash
pip install tqdm matplotlib==3.7 numpy==1.26.4 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru shapely
```

### 2. Checkpoint Download

Download SAM 2.1 checkpoints using:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

The checkpoint for Motion Predictor has been included in this repo at [`sam2/checkpoints/mp.pth`](sam2/checkpoints/mp.pth). No additional download needed.

### 3. Data Preparation

Please prepare the data following [`data/data_preparation.md`](data/data_preparation.md).

### 4. Inference & Evaluation

Run inference and evaluation on all datasets using:

```bash
bash scripts/test.sh
```

We provide our **raw results** on [Google Drive](https://drive.google.com/file/d/1loIjhCcQcjVlgbPvryaDefDE4a1tdEUS/view?usp=sharing). Download `samosa_raw_results.zip`, unzip it, and place the contents in `output/samosa_raw_results` to reproduce our reported metrics directly. You can run evaluation on prepared raw results by running:

```bash
python utils/calc_vot_metrics.py --res_path output/samosa_raw_results # LaSOT_ext, OTB100
python utils/calc_uav_metrics.py --res_path output/samosa_raw_results # anti-UAV benchmarks
```

## 🙏 Acknowledgment

SAMOSA is built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file), [SAMURAI](https://github.com/yangchris11/samurai), and [SAMITE](https://github.com/Sam1224/SAMITE). Thanks for their great work!

## 📚 Citation

If you find SAMOSA useful in your research, please consider citing our work:

```bibtex
@article{zhu2026samosa,
  title         = {Segment Anything with Motion, Geometry, and Semantic Adaptation for Complex Nonlinear Visual Object Tracking},
  author        = {Zhu, Deyi and Wang, Yuji and Liu, Yong and Tang, Yansong and Yu, Bingyao and Lu, Jiwen and Zhou, Jie},
  journal       = {arXiv preprint arXiv:2605.22538},
  year          = {2026}
}
```