# Imagine How to Change: Explicit Procedure Modeling for Change Captioning

<div align="center">

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<!-- **Jiayang Sun**$^{*1}$, **Zixin Guo**$^{*2}$, **Min Cao**$^{\dagger1}$, **Guibo Zhu**$^{\dagger3}$, **Jorma Laaksonen**$^{2}$ -->

<!-- **Published at ICLR 2026** -->

</div>

## 📢 News
* **[2026-01-26]** The paper is accepted by **ICLR 2026**! 🎉
* **[Coming Soon]** We are currently organizing the code and checkpoints. They will be released shortly. Please star ⭐ this repo to get notifications!

## 🏠 Abstract
This repository contains the official implementation of the paper **"Imagine How to Change: Explicit Procedure Modeling for Change Captioning"**.

Change captioning generates descriptions that explicitly describe the differences between two visually similar images. Existing methods operate on static image pairs, thus ignoring the rich temporal dynamics of the change procedure.

We introduce **ProCap**, a novel framework that reformulates change modeling from static image comparison to dynamic procedure modeling. 
1.  [cite_start]**Explicit Procedure Modeling:** Trains a procedure encoder to learn the change procedure from a sparse set of keyframes obtained by interpolating and sampling intermediate frames[cite: 20, 21].
2.  [cite_start]**Implicit Procedure Captioning:** Integrates the encoder within an encoder-decoder model, using learnable procedure queries to prompt the encoder for inferring the latent procedure representation[cite: 24, 25].

[cite_start]Experiments on CLEVR-Change, Spot-the-Diff, and Image-Editing-Request demonstrate the effectiveness of ProCap[cite: 27].

![ProCap Framework](assets/framework.png)
*Figure 1: The proposed two-stage ProCap framework. (Left) Explicit Procedure Modeling stage. (Right) Implicit Procedure Captioning stage.*

## 🚀 TODO List
- [x] Release the paper.
- [ ] Release the training and inference code.
- [ ] Release the pre-trained checkpoints (ProCap best models) and the processed datasets (CLEVR-Change, Spot-the-Diff, Image-Editing-Request).

## 📖 Citation
If you find our work or this repository useful, please consider citing our paper:

```bibtex
@inproceedings{
anonymous2026imagine,
  title={Imagine How To Change: Explicit Procedure Modeling for Change Captioning},
  author={Sun, Jiayang and Guo, Zixin and Cao, Min and Zhu, Guibo and Laaksonen, Jorma},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
