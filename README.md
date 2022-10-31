# Research Code for RGBDGaze

This is the research repository for RGBDGaze: Gaze Tracking on Smartphones with RGB and Depth Data (ICMI 2022).
It contains the training code and dataset link.

# Environment

- CUDA
- docker 
- docker-compose
- nvidia-docker

# How to use

## 1. Download dataset and pretrained RGB model

- Dataset: 
- Pretrained RGB model: 

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

For training RGB+D model, run
```
$ python lopo.py --config ./config/rgbd.yml
```

For training RGB model, run
```
$ python lopo.py --config ./config/rgb.yml
```

# Dataset description

# Reference

[Download the paper here.](https://rikky0611.github.io/resource/paper/rgbdgaze_icmi2022_paper.pdf)

```



```

# License


