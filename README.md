# Research Code for RGBDGaze

![image](https://user-images.githubusercontent.com/12772049/200086408-2d85ff1b-9858-480c-b972-0b4b48239906.png)


This is the research repository for [RGBDGaze: Gaze Tracking on Smartphones with RGB and Depth Data](https://dl.acm.org/doi/10.1145/3536221.3556568), presented at ACM ICMI 2022.

It contains the training code and dataset link. 

# Environment

- docker 
- docker-compose
- nvidia-docker
- nvidia-driver

# How to use

## 1. Download dataset and pretrained RGB model

- Dataset: https://www.dropbox.com/scl/fi/2ztabhldq7w0vndqmywrf/RGBDGaze_dataset.zip?rlkey=kiit9lii5lq1tjjvohv1mwl7b&dl=0
- RGB part of the Spatial CNN model pretrained with [GazeCapture](https://gazecapture.csail.mit.edu/) dataset: https://www.dropbox.com/s/dgpn0j212l260q1/pretrained_rgb.pth?dl=0

## 2. Clone

```
$ git clone https://github.com/FIGLAB/RGBDGaze
```

## 3. Setup

```
$ cp .env{.example,}
```

In `.env`, you can set a path to your data directory.

## 4. Docker build & run

```
$ DOCKER_BUILDKIT=1 docker build -t rgbdgaze --ssh default .
$ docker-compose run --rm experiment
```

## 5. Run 

Prepare following files in the docker container
- /root/datadrive/RGBDGaze/dataset/RGBDGaze_dataset
- /root/datadrive/RGBDGaze/models/SpatialWeightsCNN_gazecapture/pretrained_rgb.pth

Make tensors to be used for training
```
$ cd preprocess
$ python format.py
```

For training RGB+D model, run
```
$ python lopo.py --config ./config/rgbd.yml
```

For training RGB model, run
```
$ python lopo.py --config ./config/rgb.yml
```

# Dataset description

## Overview

The data is organized in the following manner:

- 45 participants (*1)
- synchronized RGB + Depth images for different four context
    - standing, sitting, walking, and lying
- meta data
    - corresponding gaze target on the screen
    - detected face bounding box
    - acceleration data
    - device id
    - intrinsic camera parameter of the device

- *1: We used 50 participants data in the paper. However, five of them did not agree to be included in the public dataset.


## Structure

The folder structure is organized like this:

```
RGBDGaze_dataset
│   README.txt
│   iphone_spec.csv   
│
└───P1
│   │   intrinsic.json
│   │
│   └───decoded
│       │   
│       └───standing
│       │       │   label.csv
│       │       │
│       │       └───rgb
│       │       │   1.jpg
│       │       │   2.jpg ...
│       │       │
│       │       └───depth
│       │       
│       └───sitting
│       └───walking
│       └───lying
│   
└───P2 ...
```


# Reference

[Download the paper here.](https://rikky0611.github.io/resource/paper/rgbdgaze_icmi2022_paper.pdf)

```
Riku Arakawa, Mayank Goel, Chris Harrison, Karan Ahuja. 2022. RGBDGaze: Gaze Tracking on Smartphones with RGB and Depth Data In Proceedings of the 2022 International Conference on Multimodal Interaction (ICMI '22). Association for Computing Machinery, New York, NY, USA.
```

```
@inproceedings{DBLP:conf/icmi/ArakawaG0A22,
  author    = {Riku Arakawa and
               Mayank Goel and
               Chris Harrison and
               Karan Ahuja},
  title     = {RGBDGaze: Gaze Tracking on Smartphones with {RGB} and Depth Data},
  booktitle = {International Conference on Multimodal Interaction, {ICMI} 2022, Bengaluru,
               India, November 7-11, 2022},
  pages     = {329--336},
  publisher = {{ACM}},
  year      = {2022},
  doi       = {10.1145/3536221.3556568},
  address   = {New York},
}
```

# License

GPL v 2.0 License file present in repo. Please contact innovation@cmu.edu if you would like another license for your use.
