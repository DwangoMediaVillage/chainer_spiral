# ChainerSPIRAL

A modified implementation of [Synthesizing Programs for Images using Reinforced Adversarial Learning](https://arxiv.org/abs/1804.01118) (SPIRAL) using [ChainerRL](https://github.com/chainer/chainerrl) and [MyPaint](https://github.com/mypaint/mypaint).

![](images/movie.gif)

## Dependencies

- [Pipenv](https://pipenv.readthedocs.io/en/latest/)
- [MyPaint](https://github.com/mypaint/mypaint)
- [Docker](https://www.docker.com/)

## Run pre-trained models on Docker

```
$ cd docker
$ docker build . -t chainer_spiral
$ docker run -t --name run_chainer_spiral_demo chainer_spiral pipenv run python demo.py many trained_models/mnist/64296000 result.png --without-dataset
$ docker cp run_chainer_spiral_demo:/chainer_spiral/ChainerSPIRAL/result.png .
```

If `docker cp ...` doesn't work because of a permission error, change permission of the current directory by `chmod a+rw .`

You can choose a demo mode from `static`, `many`, and `movie` (shown the above):

An example of static demo:

![](images/static.png)

Many demo:

![](images/many.png)


## How to setup manually

### Install dependencies (CentOS)

```
$ sudo yum install gcc gobject-introspection-devel json-c-devel glib2-devel git python autoconf intltool gettext libtool swig python-setuptools gettext gcc-c++ python-devel numpy gtk3-devel pygobject3-devel libpng-devel lcms2-devel json-c-devel gtk3 gobject-introspection
```

### Install libmypaint

```
$ git clone https://github.com/mypaint/libmypaint
$ cd libmypaint
$ git checkout 0c07191409bd257084d4ea7576deb832aac8868b
$ ./autogen.sh
$ ./configure --prefix=<your-installation-prefix>
$ make install
```

### Install mypaint-brushes

```
$ git clone  https://github.com/mypaint/mypaint-brushes.git
$ cd mypaint-brushes
$ git checkout 769ec941054725a195e77d8c55080344e2ab77e4
$ ./autogen.sh
$ ./configure --prefix=<your-installation-prefix>
$ make install
```

### Build MyPaint with python support

```
$ mkdir build_mypaint && cd buid_mypaint
$ git clone https://github.com/mypaint/mypaint.git
$ cd mypaint
$ git checkout 57685af8dbd65719d7874bc501094bade85d94e7
$ cd ../
$ pipenv install --python 3.6
$ pipenv install numpy pygobject pycairo
$ pipenv shell
$ cd mypaint
$ python setup.py build
$ readlink -f build/lib.linux-x86_64-3.6  # append this path to .env file
```

### Set envrionment variables

Make sure that `<your-installation-prefix>/lib` is in `LD_LIBRARY_PATH` and `PYTHONPATH`. Also `PKG_CONFIG_PATH` shoud include `<your-installation-prefix>/lib` and `<your-installation-prefix>/share`.

### Install this project's dependencies

```
$ pipenv run install
```

### Check your installation

Go to this repo's directory and run tests by `pipenv run test`

### Train model from scrach

```
pipenv run python train.py settings/default.yaml logs
```

Details of training options available on comments of `settings/default.yaml`.