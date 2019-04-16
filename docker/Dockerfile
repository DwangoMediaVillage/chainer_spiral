FROM centos:7.2.1511
MAINTAINER KazumaSasaki <kazuma_sasaki@dwango.co.jp>


# setup python3.6
RUN yum -y update
RUN yum -y install gcc git wget gcc gobject-introspection-devel json-c-devel glib2-devel git python autoconf intltool gettext libtool swig python-setuptools gettext gcc-c++ python-devel numpy gtk3-devel pygobject3-devel libpng-devel lcms2-devel json-c-devel gtk3 gobject-introspection 

RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm

RUN yum -y install python36u python36u-devel python36u-pip
RUN ln -sf /usr/bin/python3.6 /usr/bin/python3 && ln -sf /usr/bin/pip3.6 /usr/bin/pip3
RUN pip3 install -U pip
RUN pip3 install pipenv

# install libmypaint
RUN yum -y install make

RUN mkdir chainer_spiral
WORKDIR chainer_spiral

RUN git clone https://github.com/mypaint/libmypaint

WORKDIR libmypaint

RUN git checkout 0c07191409bd257084d4ea7576deb832aac8868b
RUN ./autogen.sh && ./configure && make install -j6

# install mypaint-brushes
WORKDIR /chainer_spiral

RUN git clone  https://github.com/mypaint/mypaint-brushes.git

WORKDIR mypaint-brushes

RUN git checkout 769ec941054725a195e77d8c55080344e2ab77e4
RUN ./autogen.sh && ./configure && make install -j4

# build mypaint with python support
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig:/usr/local/share/pkgconfig/

WORKDIR /chainer_spiral

RUN mkdir build_mypaint
WORKDIR build_mypaint

RUN git clone https://github.com/mypaint/mypaint.git
RUN pip3 install numpy pygobject pycairo

WORKDIR mypaint
RUN git checkout 57685af8dbd65719d7874bc501094bade85d94e7
RUN python3 setup.py build

# install dependency of nose test
RUN yum install -y python36u-test

WORKDIR /root

# add path
ENV PYTHONPATH /usr/local/lib:/chainer_spiral/build_mypaint/mypaint/build/lib.linux-x86_64-3.6:/root:$PYTHONPATH
ENV LD_LIBRARY_PATH /usr/local/lib:/chainer_spiral/build_mypaint/mypaint/build/lib.linux-x86_64-3.6:$LD_LIBRARY_PATH

# avoid segmentation fault when importing matplotlib
RUN yum -y install libSM.x86_64

RUN pip3 install cached-property==1.5.1 \
                certifi==2019.3.9 \
                chainerrl==0.6.0 \
                chardet==3.0.4 \
                cycler==0.10.0 \
                filelock==3.0.10 \
                future==0.17.1 \
                git+https://github.com/chainer/chainer@4cb03dfb1d201a708ce5267126a4df89b799a211#egg=chainer \
                gym==0.12.1 \
                idna==2.8 \
                kiwisolver==1.0.1 \
                matplotlib==3.0.3 \
                nose==1.3.7 \
                numpy==1.16.2 \
                opencv-python==4.1.0.25 \
                pandas==0.24.2 \
                pillow==6.0.0 \
                protobuf==3.7.1 \
                pycairo==1.18.0 \
                pyglet==1.3.2 \
                pygobject==3.32.0 \
                pyparsing==2.4.0 \
                python-dateutil==2.8.0 \
                pytz==2019.1 \
                pyyaml==5.1 \
                rdp==0.8 \
                requests==2.21.0 \
                scipy==1.2.1 \
                six==1.12.0 \
                urllib3==1.24.1
