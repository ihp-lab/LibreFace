
  

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

<p align="center">
  <img src="https://github.com/ihp-lab/LibreFace/blob/main/media/System.png" width="350px" />
</p>

## Getting started with Python installation

### Dependencies

- Python 3.8
- You should have `cmake` installed in your system.
    - **For Linux users** - `sudo apt-get install cmake`. If you run into trouble, consider upgrading to the latest version ([instructions](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line)).
    - **For Mac users** - `brew install cmake`.


### Installation

You can first create a new Python 3.8 environment using `conda` and then install this package using `pip` from the PyPI hub:

```console
conda create -n libreface_env python=3.8
conda activate libreface_env
pip install --upgrade libreface
```

### Usage

#### Using commandline

You can use this package through the command line using the following command.
```console
libreface --input_path="path/to/your_image_or_video"
```

Note that the above script would save results in a CSV at the default location - `sample_results.csv`. If you want to specify your own path, use the `--output_path`  command line argument,
```console
libreface --input_path="path/to/your_image_or_video" --output_path="path/to/save_results.csv"
```

To change the device used for inference, use the `--device` command line argument,
```console
libreface --input_path="path/to/your_image_or_video" --device="cuda:0"
```

To save intermediate files, `libreface` uses a temporary directory that defaults to `./tmp`. To change the temporary directory path,
```console
libreface --input_path="path/to/your_image_or_video" --temp="your/temp/path"
```

For video inference, our code processes the frames of your video in batches. You can specify the batch size and the number of workers for data loading as follows,
```console
libreface --input_path="path/to/your_video" --batch_size=256 --num_workers=2 --device="cuda:0"
```

Note that by default, the `--batch_size` argument is 256, and `--num_workers` argument is 2. You can increase or decrease these values according to your machine's capacity.

**Examples**

Download a [sample image](https://github.com/ihp-lab/LibreFace/blob/main/examples/sample_disfa.png) from our GitHub repository. To get the facial attributes for this image and save to a CSV file, simply run,
```console
libreface --input_path="sample_disfa.png"
```

Download a [sample video](https://github.com/ihp-lab/LibreFace/blob/main/examples/sample_disfa.avi) from our GitHub repository. To run the inference on this video using a GPU and save the results to `my_custom_file.csv` run the following command,
```console
libreface --input_path="sample_disfa.avi" --output_path="my_custom_file.csv" --device="cuda:0"
```

Note that for videos, each row in the saved CSV file corresponds to individual frames in the given video.

#### Using Python script

Here’s how to use this package in your Python scripts.

To assign the results to a Python variable,
```python
import libreface 
detected_attributes = libreface.get_facial_attributes(image_or_video_path)
```

To save the results to a csv file, use the `output_save_path` parameter,
```python
import libreface 
libreface.get_facial_attributes(image_or_video_path,
                                output_save_path = "your_save_path.csv")
```

To change the device used for inference, use the `device` parameter,
```python

import libreface 
libreface.get_facial_attributes(image_or_video_path,
                                device = "cuda:0") # can be "cpu" or "cuda:0", "cuda:1", ...
```

To save intermediate files, libreface uses a temporary directory that defaults to `./tmp`. To change the temporary directory path, use the `temp_dir` parameter,
```python
import libreface 
libreface.get_facial_attributes(image_or_video_path,
                                temp_dir = "your/temp/path")
```

For video inference, our code processes the frames of your video in batches. You can specify the batch size and the number of workers for data loading as follows, 
```python
import libreface 
libreface.get_facial_attributes(video_path,
                                batch_size = 256,
                                num_workers = 2)
```

Note that by default, the `batch_size` is 256, and `num_workers` is 2. You can increase or decrease these values according to your machine's capacity.

Weights of the model are automatically downloaded at `./libreface_weights/` directory. If you want to download and save the weights to a separate directory, please specify the parent folder for weights using the `weights_download_dir` as follows,
```python
import libreface 
libreface.get_facial_attributes(image_or_video_path,
                                weights_download_dir = "your/directory/path")
```
 

## Getting Started with Derivative Tools (New 2.0 Models Available! Recommended)

We offer several derivative tools on the .NET platform to facilitate easier integration of LibreFace into various systems, in addition to pytorch code. These works are based on ONNX platform weights exported from the model weights of this project.

<p align="center">
  <img src="https://github.com/ihp-lab/LibreFace/blob/main/media/software.png" />
</p>

+ NuGet Package: We have released a [NuGet Package named `LibreFace`](https://www.nuget.org/packages/LibreFace). This NuGet Package contains the ONNX weight files, and its source code is located in [this directory of the OpenSense repository](https://github.com/ihp-lab/OpenSense/tree/master/Utilities/LibreFace). For how to integrate it, you can refer to the documentation that comes with the Package or the source code of the OpenSense Component below. This Package is cross-platform compatible and is recommended to be used with an ONNX Runtime Execution Provider with acceleration features. For non-.NET developers, you can access the ONNX weight files inside the package by changing the extension from `.nupkg` to `.zip`.

+ OpenSense Component: We have also wrapped LibreFace into a component available in OpenSense. With this work, other components in OpenSense can be used in conjunction with LibreFace in Zero-Code setup for real-time or non-real-time inference. Its source code is mainly stored in [this directory of the OpenSense repository](https://github.com/ihp-lab/OpenSense/tree/master/Components/LibreFace). Running this component by default requires CUDA support, but other ONNX Providers can be used when compiling from the source code. A OpenSense Pipeline we were using for testing can be used as an example and can be downloaded from [here](https://github.com/ihp-lab/OpenSense/releases/download/3.2.0/20230825__LibreFace__Injector__AzureKinect.pipe.json). Please set the camera you want to use before running it.

+ Command Line Tool: For the common scenario of analyzing videos and exporting results to text files, we have created a dedicated command-line tool for batch processing of video files. This tool can be downloaded as a compiled program from [the OpenSense release Google Drive directory](https://drive.google.com/drive/folders/1rYypeKELnva0-MGQvNJ45cgsrgjfowHw?usp=sharing). Please select a Zip file having `LibreFace Console Application` in its name to download. Its executable is called `LibreFace.App.Consoles.exe`, please run it in a command line environment as it is a command line application. The source code can be found in [this directory of the OpenSense repository](https://github.com/ihp-lab/OpenSense/tree/master/Derivatives/LibreFace.App.Consoles). It takes video files as input, and outputs one JSON file per video containing individual results of frames. For specific usage methods and running environment requirements, please refer to the documentation built into the tool. Currently, it only supports Windows and CUDA is mandatory. We are adding functionality to batch process images, and potentially adding support for other operating systems.

<p align="center">
  <img src="https://github.com/ihp-lab/LibreFace/blob/main/media/ConsoleApplication.png" alt="A screenshot of LibreFace Console Application, showing it built-in documentation." width="480px" />
</p>
  
## Training models using Python

Clone repo:

```
git clone https://github.com/ihp-lab/LibreFace.git
cd LibreFace
```

The code is tested with Python == 3.7, PyTorch == 1.10.1 and CUDA == 11.3 on NVIDIA GeForce RTX 3090. We recommend you to use [anaconda](https://www.anaconda.com/) to manage dependencies. You may need to change the torch and cuda version in the `requirements.txt` according to your computer.

```
conda create -n libreface python=3.7
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda activate libreface
pip install -r requirements.txt
```

### Facial Landmark/Mesh Detection and Alignment

As described in our paper, we first pre-process the input image by mediapipe to obatain facial landmark and mesh. The detected landmark are used to calculate the corresponding positions of the eyes and mouth in the image. We finally use these positions to align the images and center the face area.

To process the image, simpy run following commad:

```jsx
python detect_mediapipe.py
```

### AU Intensity Estimation

#### DISFA

Download the [DISFA](http://mohammadmahoor.com/disfa/) dataset from the official website here. Please be reminded that the original format of the dataset are video sequences, you need to manually process them into image frames.

Download the original video provided by DISFA. Extract it and put it under the folder `data/DISFA`.

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

#### Training/Inference

```
cd AU_Recognition
bash train.sh
bash inference.sh
```

### AU Detection

#### BP4D

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

#### Training/Inference

```
cd AU_Detection
bash train.sh
bash inference.sh
```

### Facial Expression Recognition

#### AffectNet

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

#### Training/Inference

```
cd Facial_Expression_Recognition
bash train.sh
bash inference.sh
```

### Configure

There are several options of flags at the beginning of each train/inference files. Several key options are explained below. Other options are self-explanatory in the codes. Before running our codes, you may need to change the `device`, `data_root` , `ckpt_path` , `data` and `fold`.

- `ckpt_path` A relative or absolute folder path for writing checkpoints.
- `data_root` The path to your dataset on your local machine.
- `device` Specify cuda or cpu. 
- `data` Dataset to be used. [“DSIFA”,“BP4D”,"AffectNet","RAF-DB"]
- `fold` We use five-fold cross-validation to report performance on DISFA and three-fold cross-validation on BP4D. ["0","1","2","3","4"]
- `train_csv` Training csv file to be parsed. 
- `test_csv` Testing csv file to be parsed. 
- `fm_distillation` Use feature matching distillation for training. 

## Results and Accuracy

We performed a variety of evaluations across several demographic axes to observe how our model performs on different groups of people.

### FACES Accuracy

This is the accuracy of the model when performing on the [FACES](https://faces.mpdl.mpg.de/imeji/) dataset, filtered to only include images in the dataset that were deemed correctly labelled by a majority of human raters. This dataset features a variety of ages, and the evaluation particularly denotes the model's performance on elderly people.
<p align="center">
  <img src="https://github.com/ihp-lab/LibreFace/blob/main/media/facesAccuracy.JPG" alt="Faces Accuracy Results" width="480px" />
</p>

### CAFE Accuracy
This is the accuracy of the model when performing on the [CAFE](https://nyu.databrary.org/volume/30) dataset. This dataset features faces of children across a variety of racial groups. The children in the CAFE dataset are notably younger than those in the FACES dataset, denoting the model's performance on ages 32.5 mos–8.7 yrs and across differnt racial groups. AA = African American, AS = Asian, EA = European American, LA = Latino, PI = Pacific Islander, SA = South Asian

The two columns at the bottom compare human raters on this entire dataset against the performance of our model across each emotion.
<p align="center">
  <img src="https://github.com/ihp-lab/LibreFace/blob/main/media/cafeAccuracy.JPG" alt="Cafe Accuracy Results" width="480px" />
</p>

## TODO:
Package
- [ ] Optimize facial alignment using mediapipe face aligner.
- [ ] Add option for the user to choose which model to run.
- [ ] Add multiple file inference by passing a list.
- [x] Add batch inference for video files.
- [x] Release pip package.

Validation
- [ ]  Upload Training/Validation Split csv files and CSV creation python code for model training
- [ ]  Upload Facial Expression Recognition code on RAF-DB Dataset

## License

Our code is distributed under the USC research license. See `LICENSE.txt` file for more information.

## Citation

```
@inproceedings{chang2023libreface,
      title={LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis}, 
      author={Di Chang and Yufeng Yin and Zongjian Li and Minh Tran and Mohammad Soleymani},
      year={2024},
      booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      month     = {January},
      note = {To appear}}
```

## Contact

If you have any questions, please raise an issue or email to Di Chang (`dchang@ict.usc.edu`or `dichang@usc.edu`). For issues related to the python package, write to Ashutosh Chaubey (`achaubey@usc.edu` or `achaubey@ict.usc.edu`).

## Acknowledgments

Our code follows several awesome repositories. We appreciate them for making their codes available to public.

- [KD_SRRL](https://github.com/jingyang2017/KD_SRRL)
- [XNorm](https://github.com/ihp-lab/XNorm)
- This work is sponsored by the U.S. Army Research Laboratory (ARL) under contract number W911NF-14-D-0005. The content of the information does not necessarily reflect the position or the policy of the Government, and no official endorsement should be inferred.