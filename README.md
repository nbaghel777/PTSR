# PTSR: Patch Translator for Image Super-Resolution
[Neeraj Baghel](https://sites.google.com/view/nbaghel777) , [Satish Singh](https://cvbl.iiita.ac.in/sks/) and [Shiv Ram Dubey](https://profile.iiita.ac.in/srdubey/)
<!--
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)]()
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)]()
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)]()
[![Summary](https://img.shields.io/badge/Summary-Slide-87CEEB)]()
 -->
#### News
<!--
- **April 4, 2022:** Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/swzamir/Restormer)
- **March 30, 2022:** Added Colab Demo. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C2818h7KnjNv4R1sabe14_AYL7lWhmu6?usp=sharing)
- **March 29, 2022:** Restormer is selected for an ORAL presentation at CVPR 2022 :dizzy:
- **March 10, 2022:** Training codes are released :fire:
- **March 3, 2022:** Paper accepted at CVPR 2022 :tada: 
 -->
- **Jan, 2023:** Codes are released!
- **April, 2022:** Paper submitted

<hr />

> **Abstract:** Image super-resolution generation aims to generate a high-resolution image from its low-resolution image. However, more complex neural networks bring high computational costs and memory storage. It is still an active area for offering the promise of overcoming resolution limitations in many applications. In recent years, transformers have made significant progress in computer vision tasks as their robust self-attention mechanism. However, recent works on the transformer for image super-resolution also contain convolution operations. We propose a patch translator for image super-resolution (PTSR) to address this problem. The proposed PTSR is a transformer-based GAN network with no convolution operation. We introduce a novel patch translator module for regenerating the improved patches utilising multi-head attention, which is further utilised by the generator to generate the $2\times$ and $4\times$ super-resolution images. The experiments are performed using benchmark datasets, including DIV2K, Set5, Set14, and BSD100. The results of the proposed model is improved on an average for $4\times$ super-resolution by 21.66\% in PNSR score and 11.59\% in SSIM score, as compared to the best competitive models. We also analyse the proposed loss and saliency map to show the effectiveness of the proposed method. The code used in the paper will be made publicly available at https://github.com/nbaghel777/PTSR.
<hr />

# Network Architecture
<img src = "https://github.com/nbaghel777/PTSR/blob/main/VTrans-VisionTranslator.jpg"> 

# Training and Evaluation

<img src = "https://github.com/nbaghel777/SRTransGAN/blob/main/result2.png"> 


# Results
<img src = "https://github.com/nbaghel777/SRTransGAN/blob/main/result1.png"> 

# Contact:
Should you have any question, please contact neerajbaghel@ieee.org
1) SRTransGAN: SRTransGAN: Image Super-Resolution using Transformer based Generative Adversarial Network

# Citation
If you use SRTransGAN, please consider citing:

@inproceedings{baghel2022srtransgan,
    title={SRTransGAN: Image Super-Resolution using Transformer based Generative Adversarial Network}, 
    author={Neeraj Baghel and Satish Singh and Shiv Ram Dubey},
    year={2022}
}

# Related Works: 
1) Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)
2) ViTGAN: Training GANs with Vision Transformers

