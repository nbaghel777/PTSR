# SRTransGAN: Image Super-Resolution using Transformer based Generative Adversarial Network
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

> **Abstract:** Image super-resolution aims to synthesize high-resolution image from a low-resolution image. 
It is an active area to overcome the resolution limitations in several applications like low-resolution object-recognition, medical image enhancement, etc. 
The generative adversarial network (GAN) based methods have been the state-of-the-art for image super-resolution by utilizing the convolutional neural networks (CNNs) based generator and discriminator networks. However, the CNNs are not able to exploit the global information very effectively in contrast to the transformers, which are the recent breakthrough in deep learning by exploiting the self-attention mechanism. Motivated from the success of transformers in language and vision applications, we propose a SRTransGAN for image super-resolution using transformer based GAN. Specifically, we propose a novel transformer-based encoder-decoder network as a generator to generate $2\times$ images and $4\times$ images. We design the discriminator network using vision transformer which uses the image as sequence of patches and hence useful for binary classification between synthesized and real high-resolution images. 
The proposed SRTransGAN outperforms the existing methods by 4.38\% on an average of PSNR and SSIM scores. We also analyze the saliency map to show the effectiveness of the proposed method. The code used in the paper will be publicly available at [https://github.com/nbaghel777/SRTransGAN](https://github.com/nbaghel777/SRTransGAN).
<hr />

# Network Architecture
<img src = "https://github.com/nbaghel777/SRTransGAN/blob/main/SRTransG.png"> 

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

