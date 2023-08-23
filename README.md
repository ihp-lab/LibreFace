

<div align="center">
  <h1 align="center">LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis</h1>

  <p align="center">

<a href="https://boese0601.github.io/">
    Di Chang</a>,
<a href="https://yufengyin.github.io/">
    Yufeng Yin</a>,
    Zongjian Li,
<a href="https://scholar.google.com/citations?user=HuuQRj4AAAAJ&hl=en">
    Minh Tran</a>,
<a href="https://people.ict.usc.edu/~soleymani/">
    Mohammad Soleymani</a>

<br>
                    
<a href="https://ict.usc.edu/">Institute for Creative Technologies, University of Southern California</a>
                    

<strong>WACV 2024</strong>
<br />
<a href="https://arxiv.org/abs/2308.10713">Arxiv</a> | <a href="https://boese0601.github.io/libreface">Project page</a>
<br />
</p>
</div>


## Introduction

This is the official implementation of our WACV 2024 Application Track paper: LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis. LibreFace is an open-source and comprehensive toolkit for accurate and real-time facial expression analysis with both CPU-only and GPU-acceleration versions. LibreFace eliminates the gap between cutting-edge research and an easy and free-to-use non-commercial toolbox. We propose to adaptively pre-train the vision encoders with various face datasets and then distillate them to a lightweight ResNet-18 model in a feature-wise matching manner. We conduct extensive experiments of pre-training and distillation to demonstrate that our proposed pipeline achieves comparable results to state-of-the-art works while maintaining real-time efficiency. LibreFace system supports cross-platform running, and the code is open-sourced in C# (model inference and checkpoints) and Python (model training, inference, and checkpoints).

![media/Comparison.jpg](media/Comparison.jpg)

## Installation

Clone repo:

```
git clone https://github.com/ihp-lab/LibreFace.git
cd LibreFace
```

The code is tested with Python == 3.7, PyTorch == 1.10.1 and CUDA == 11.3 on NVIDIA GeForce RTX 3090. We recommend you to use [anaconda](https://www.anaconda.com/) to manage dependencies. You may need to change the torch and cuda version in the `requirements.txt` according to your computer.

```
conda create -n libreface python=3.7
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda activate libreface
pip install -r requirements.txt
```

## Facial Landmark/Mesh Detection and Alignment

As described in our paper, we first pre-process the input image by mediapipe to obatain facial landmark and mesh. The detected landmark are used to calculate the corresponding positions of the eyes and mouth in the image. We finally use these positions to align the images and center the face area.

To process the image, simpy run following commad:

```jsx
python detect_mediapipe.py
```

## AU Intensity Estimation

### DISFA

Download the [DISFA](http://mohammadmahoor.com/disfa/) dataset from the official website here. Please be reminded that the original format of the dataset are video sequences, you need to manually process them into image frames.

Download the original video provided by DISFA+. Extract it and put it under the folder `data/DISFA`.

Preprocess the images by previous mediapipe script and you should get a dataset folder like below:

```
data
├── DISFA
│ ├── images
│ ├── landmarks
│ └── aligned_images
├── BP4D
├── AffectNet
└── RAF-DB
```

### Training/Inference

```
cd AU_Recognition
bash train.sh
bash inference.sh
```

## AU Detection

### BP4D

Download the [BP4D](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) dataset from the official website. Extract it and put it under the folder `data/BP4D`.

Preprocess the images by previous mediapipe script and you should get a dataset folder like below:

```
data
├── DISFA
│ ├── images
│ ├── landmarks
│ └── aligned_images
├── BP4D
│ ├── images
│ ├── landmarks
│ └── aligned_images
├── AffectNet
└── RAF-DB
```

### Training/Inference

```
cd AU_Detection
bash train.sh
bash inference.sh
```

## Facial Expression Recognition

### AffectNet

Download the [AffectNet](http://mohammadmahoor.com/affectnet/) dataset from the official website. Extract it and put it under the folder `data/AffectNet`.

Preprocess the images by previous mediapipe script and you should get a dataset folder like below:

```
data
├── DISFA
│ ├── images
│ ├── landmarks
│ └── aligned_images
├── BP4D
│ ├── images
│ ├── landmarks
│ └── aligned_images
├── AffectNet
│ ├── images
│ ├── landmarks
│ └── aligned_images
└── RAF-DB
```

### Training/Inference

```
cd Facial_Expression_Recognition
bash train.sh
bash inference.sh
```

## Configure

There are several options of flags at the beginning of each train/inference files. Several key options are explained below. Other options are self-explanatory in the codes. Before running our codes, you may need to change the `true_gpu`, `trainpath/testpath` , `logdir`and `loadckpt` (only for testing).

- `logdir` A relative or absolute folder path for writing logs.
- `true_gpu` The true GPU IDs, used for setting CUDA_VISIBLE_DEVICES in the code. You may change it to your GPU IDs.
- `gpu` The GPU ID used in your experiment. If true_gpu: “5, 6”. Then you could use gpu: [0], gpu: [1], or gpu: [0, 1]
- `loadckpt` The checkpoint file path used for testing.
- `trainpath/testpath` A relative or absolute folder path for training or testing data. You may need to change it to your data folder.
- `outdir` A relative or absolute folder path for generating depth maps and writing point clouds(DTU).
- `plydir` A relative or absolute folder path for writing point clouds(Tanks).
- `dataset` Dataset to be used. [“dtu_train”,“dtu_test”,“tanks”]
- `resume` Resume training from the latest history.

## TODO:

- [ ]  Upload Training/Validation Split csv files for model training
- [ ]  Upload Facial Expression Recognition code on RAF-DB Dataset
- [ ]  Open-Source C# code for LibreFace

## License

Our code is distributed under the MIT License. See `LICENSE` file for more information.

## Citation

```
@inproceedings{chang2022rc,  title={RC-MVSNet: Unsupervised Multi-View Stereo with Neural Rendering},  author={Chang, Di and Bo{\v{z}}i{\v{c}}, Alja{\v{z}} and Zhang, Tong and Yan, Qingsong and Chen, Yingcong and S{\"u}sstrunk, Sabine and Nie{\ss}ner, Matthias},  booktitle={Proceedings of the European conference on computer vision (ECCV)},  year={2022}}
```

## Contact

If you have any questions, please raise an issue or email to Di Chang (`dchang@ict.usc.edu`or `dichang@usc.edu`).

## Acknowledgments

Our code follows several awesome repositories. We appreciate them for making their codes available to public.

- [KD_SRRL](https://github.com/jingyang2017/KD_SRRL)
- [XNorm](https://github.com/ihp-lab/XNorm)
