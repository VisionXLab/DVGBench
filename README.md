<div align="center">
<h1>DVGBench: Implicit-to-Explicit Visual Grounding Benchmark in UAV Imagery with Large Vision-Language Models</h1>

<div>
    <a href='https://zytx121.github.io/' target='_blank'>Yue Zhou</a><sup>1</sup>&emsp;
    Jue Chen<sup>1</sup>&emsp;
    Zilun Zhang<sup>2</sup>&emsp;
    Penghui Huang<sup>3</sup>&emsp;
    Ran Ding<sup>3</sup>&emsp;
    Zhentao Zou<sup>3</sup>&emsp;
    PengFei Gao<sup>4</sup>&emsp;
    Yuchen Wei<sup>4</sup>&emsp;
    Ke Li<sup>4</sup>&emsp;
    <a href='https://yangxue.site/' target='_blank'>Xue Yang</a><sup>3</sup>&emsp;
    <a href='https://ee.sjtu.edu.cn/FacultyDetail.aspx?id=53&infoid=66' target='_blank'>Jiang Xue</a><sup>3</sup>&emsp;
    Hongxin Yang<sup>1</sup>&emsp;
    Jonathan Li<sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>East China Normal University&emsp; 
    <sup>2</sup>Zhejiang University&emsp; 
    <sup>3</sup>Shanghai Jiao Tong University&emsp;
    <sup>4</sup>Information Engineering University&emsp;
</div>

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/xxx)
[![Paper](https://img.shields.io/badge/ISPRS_JPRS-Paper-orange.svg)](https://xxx)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/erenzhou/DVGBench)

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

</div>

<p align="center">
    <img src="images/demo1.gif" width=60%>
</p>


---

## 📢 Latest Updates

- **[2026.01.03]** We will release the benchmark and evaluation code ASAP.
- **[2026.01.02]** Accepted by ISPRS JPRS.

---

## Abstract

*Remote sensing (RS) large vision–language models (LVLMs) have shown strong promise across visual grounding (VG) tasks. However, existing RS VG datasets predominantly rely on explicit referring expressions—such as relative position, relative size, and color cues—thereby constraining performance on implicit VG tasks that require scenario-specific domain knowledge. This article introduces \dataset, a high-quality implicit VG benchmark for drones, covering six major application scenarios: traffic, disaster, security, sport, social activity, and productive activity. Each object provides both explicit and implicit queries. Based on the dataset, we design DroneVG-R1, an LVLM that integrates the novel Implicit-to-Explicit Chain-of-Thought (I2E-CoT) within a reinforcement learning paradigm. This enables the model to take advantage of scene-specific expertise, converting implicit references into explicit ones and thus reducing grounding difficulty. Finally, an evaluation of mainstream models on both explicit and implicit VG tasks reveals substantial limitations in their reasoning capabilities. These findings provide actionable insights for advancing the reasoning capacity of LVLMs for drone-based agents.*

<div align="center">
  <img src="images/Idea.png" width=50%>
  <div style="display: inline-block; color: #999; padding: 2px;">
      Infants can understand references composed of colors and relative positions easily, but cannot comprehend references involving common sense or domain knowledge. We refer to the latter as Implicit Visual Grounding.
  </div>
</div>

---

## 🏆 Contributions

- **Benchmark:** We introduce DVGBench, a human-annotated VG benchmark designed for real-world UAV applications, is presented. It spans six diverse scenarios and provides both box and mask-level annotations, along with explicit as well as implicit referring expressions.

- **Model:**  Based on DVGBench, DroneVG-R1, an LVLM tailored for implicit VG in UAV contexts, is proposed. A segmentation model is incorporated to support reasoning segmentation.

- **Method:**  An I2E-CoT strategy is introduced to enhance grounding accuracy by converting implicit references into explicit textual descriptions. To incentivize this conversion, a novel reasoning reward function based on explicit reference similarity is designed.

- **Exploration:** Extensive evaluations of existing models are performed, uncovering their limitations in implicit VG. Through comparative analysis of performance on explicit versus implicit queries, insights into the reasoning gaps and directions for improvement are provided.

---

## 💬 Benchmark

Examples of six UAV application scenarios in DVGBench

<p align="center">
  <img src="images/DVGBench.png" width=100%>
</p>

Please download the [dataset](https://huggingface.co/datasets/erenzhou/DVGBench) first and then refer to the code in the evaluation to infer and evaluate the score.

---

## 🤖 DroneVG-R1

Framework of the DroneVG-R1, which comprises a reasoning model and a segmentation model. The reasoning model
is an LVLM that generates reasoning chains and provides box-level results. Subsequently, the segmentation model produces a
pixel-wise mask based on the box. In addition to regular format rewards and perceptual rewards, we have also designed a reasoning
reward to enhance the quality of the model’s implicit-to-explicit conversion through human-annotated explicit references.

<p align="center">
  <img src="images/DroneVG-R1.png" width=100%>
</p>

<p align="center">
  <img src="images/analysis.png" width=100%>
</p>

<p align="center">
  <img src="images/analysis3.png" width=60%>
</p>


---

## 🔍 I2E-CoT

Overview of the Implicit-to-Explicit mechanism. This diagram compares the standard Group Relative Policy Optimization (GRPO) with our I2E-CoT approach. The GRPO mislocates the left-turning vehicle due to visual attention distraction during reasoning. In contrast, the I2E-CoT method employs the \<explicit\> token to generate an explicit reference for the object, correcting the initial localization and producing the correct answer. Attention graphs reveal that during the \<explicit\> phase, I2E-CoT identifies the explicit "green" cue, substantially increasing attention to the corresponding image tokens (blue line).

<p align="center">
  <img src="images/analysis2.png" width=100%>
</p>


---

## 🚀 Exploration

Two examples used to demonstrate the impact of the I2E-CoT mechanism on image attention. In the sports scenario example on the right, we observed a moment of shift in image attention. Before the phrase "white shirt" appeared, the model's attention was somewhat dispersed or focused on the left person. After the phrase "white shirt" appeared, as shown in the attention proportion curves and heatmaps, the model’s image attention shifted significantly from the left person to the rider on the right wearing a white shirt. It was this single explicit descriptive word that refined the model’s localization, making the attention accurately lock onto the target rider. 

<p align="center">
  <img src="images/analysis4.png" width=100%>
</p>


## 🎬 Demo

<p align="center">
    <img src="images/demo2.gif" width=60%>
</p>
<p align="center">
    <img src="images/demo3.gif" width=60%>
</p>
<p align="center">
    <img src="images/demo4.gif" width=60%>
</p>
<p align="center">
    <img src="images/demo5.gif" width=60%>
</p>


## 📜 Citation
```bibtex
@article{zhou2026dvgbench,
  author={Zhou, Yue and Chen, Jue and Huang, Penghui and Ding, Ran and Zou, Zhentao and Gao, Pengfei and Li, Ke and Yang, Xue and Jiang, Xue and Yang, Hongxin and Li Jonathan},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing}, 
  title={DVGBench: Implicit-to-Explicit Visual Grounding Benchmark in UAV Imagery with Large Vision-Language Models}, 
  year={2026},
  volume={},
  number={},
  pages={}
}
```

## Acknowledgement

This implementation is based on [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL), [ms-swift](https://github.com/modelscope/ms-swift) and [Look-Back](https://github.com/PKU-YuanGroup/Look-Back). Thanks for the awesome work.


## Contact
If you have any questions, please feel free to reach out at `yzhou@geoai.ecnu.edu.cn`.