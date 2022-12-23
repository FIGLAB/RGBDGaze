ARG CUDA=10.2
ARG CUDNN=7
ARG UBUNTU=18.04

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
ENV DEBIAN_FRONTEND noninteractive

ARG CUDA
ARG CUDNN
ARG UBUNTU
ARG PYTHON=3.8.7

ENV PYTHON_ROOT /root/local/python-$PYTHON
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT /root/.pyenv
ENV POETRY=1.2.1

RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    less \
    libbz2-dev \
    libffi-dev \
    libgl1 \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    llvm \
    make \
    openssh-client \
    python-openssl \
    tk-dev \
    tmux \
    unzip \
    vim \
    wget \
    xz-utils \
    zip \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python build
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    $PYENV_ROOT/plugins/python-build/install.sh && \
    /usr/local/bin/python-build -v $PYTHON $PYTHON_ROOT && \
    rm -rf $PYENV_ROOT

ENV HOME /root
WORKDIR $HOME

# install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY python3 -
ENV PATH $HOME/.local/bin:$PATH

COPY pyproject.toml poetry.lock poetry.toml $WORKDIR/


RUN mkdir -m 700 $HOME/.ssh && ssh-keyscan github.com > $HOME/.ssh/known_hosts
RUN --mount=type=ssh poetry install --no-root
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

