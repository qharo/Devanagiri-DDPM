# Devanagiri-DDPM
A PyTorch implementation of [Denoising Diffusion Probabilistic Methods](https://arxiv.org/pdf/2006.11239) trained to generate [Devanagiri](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset) characters.

### Quickstart

#### Installation
Using a Python environment of your choice,
1. ```git clone https://github.com/qharo/Devanagiri-DDPM.git```
2. ```cd Devanagiri-DDPM```
3. ```pip install -r requirements.txt```

#### Usage
To train,
```python3 train.py```

To sample,
```python3 sample.py```

The script defaults to downloading the dataset, so should you decide to train on another dataset you can edit ```dataset.py```.

### Features
<a href="https://github.com/qharo/Devanagiri-DDPM/blob/main/result_images/x0_999.png">
   <img alt="t=999" src="https://github.com/qharo/Devanagiri-DDPM/blob/main/result_images/x0_999.png"
   width="300">
</a><a href="https://github.com/qharo/Devanagiri-DDPM/blob/main/result_images/x0_499.png">
   <img alt="t=999" src="https://github.com/qharo/Devanagiri-DDPM/blob/main/result_images/x0_499.png"
   width="300">
</a>
<a href="https://github.com/qharo/Devanagiri-DDPM/blob/main/result_images/x0_374.png">
   <img alt="t=999" src="https://github.com/qharo/Devanagiri-DDPM/blob/main/result_images/x0_374.png"
   width="300">
</a>

The U-Net has 10,fadsfsd parameters, trained on 96,000 32x32 images for 40 epochs using an RTX A4000. 


### Citations and Acknowledgements
```
@misc{ho2020denoisingdiffusionprobabilisticmodels,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.11239}, 
}
```
```
@misc{nichol2021improveddenoisingdiffusionprobabilistic,
      title={Improved Denoising Diffusion Probabilistic Models}, 
      author={Alex Nichol and Prafulla Dhariwal},
      year={2021},
      eprint={2102.09672},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2102.09672}, 
}
```
* [Calvin Luo](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) and [Explaining-AI](https://www.youtube.com/watch?v=H45lF4sUgiE) for providing clarity for the math
* Inspired by [Explaining-AI's](https://github.com/explainingai-code/DDPM-Pytorch/tree/main)
