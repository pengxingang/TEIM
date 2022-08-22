FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN echo 'Building docker'

COPY ./requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
COPY . /opt/app

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# RUN conda config --set show_channel_urls yes
RUN conda install -c bioconda anarci
RUN conda install pytorch==1.10.0 cudatoolkit=10.2 -c pytorch

# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --upgrade pip
RUN pip install biopython==1.79
RUN pip install pytorch-lightning==1.6.4
RUN pip install scikit-learn==1.1.1
RUN pip install scipy==1.8.1
RUN pip install tqdm==4.64.0
RUN pip install pandas==1.4.2
RUN pip install numpy==1.22.4
RUN pip install setuptools==59.5.0
