FROM python:3.6
MAINTAINER KazumaSasaki <kazuma_sasaki@dwango.co.jp>

# install dependencies
RUN apt-get update

# add repo of git-lfs 
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash  # required to install git-lfs
RUN apt-get install -y git swig python-setuptools gettext g++ python-dev python-numpy libgtk-3-dev python-gi-dev libpng-dev liblcms2-dev libjson-c-dev gir1.2-gtk-3.0 python-gi-cairo intltool libtool git-lfs ffmpeg

# create working directory
RUN mkdir chainer_spiral
WORKDIR chainer_spiral

# clone chainer spiral repo
RUN git clone https://github.com/DwangoMediaVillage/chainer_spiral.git

# install libmypaint
RUN git clone https://github.com/mypaint/libmypaint

WORKDIR libmypaint

RUN git checkout 0c07191409bd257084d4ea7576deb832aac8868b
RUN ./autogen.sh && ./configure && make install -j4

# install mypaint-brushes
WORKDIR /chainer_spiral

RUN git clone  https://github.com/mypaint/mypaint-brushes.git

WORKDIR mypaint-brushes

RUN git checkout 769ec941054725a195e77d8c55080344e2ab77e4
RUN ./autogen.sh && ./configure && make install -j4

# setup pipenv
RUN pip install pipenv

# build mypaint with python support
WORKDIR /chainer_spiral

RUN mkdir build_mypaint
WORKDIR build_mypaint

RUN pipenv install

RUN apt-get install -y libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0

RUN pipenv install numpy pygobject pycairo --skip-lock

RUN git clone https://github.com/mypaint/mypaint.git

WORKDIR mypaint
RUN git checkout 57685af8dbd65719d7874bc501094bade85d94e7

WORKDIR /chainer_spiral/build_mypaint

RUN cd mypaint && pipenv run python setup.py build

# install python dependencies of chainer spiral
WORKDIR /chainer_spiral/ChainerSPIRAL

ENV PYTHONPATH /usr/local/lib:/chainer_spiral/build_mypaint/mypaint/build/lib.linux-x86_64-3.6:/root:$PYTHONPATH
ENV LD_LIBRARY_PATH /usr/local/lib:/chainer_spiral/build_mypaint/mypaint/build/lib.linux-x86_64-3.6:$LD_LIBRARY_PATH
ENV CHAINER_DATASET_ROOT /root  # default place to download mnist dataset

RUN pipenv install
