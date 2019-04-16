# ChainerSPIRAL

Chainer implementation of [Synthesizing Programs for Images using Reinforced Adversarial Learning](https://arxiv.org/abs/1804.01118) (SPIRAL).

## Dependencies

- [Pipenv](https://pipenv.readthedocs.io/en/latest/)
- [MyPaint](https://github.com/mypaint/mypaint)

## How to start

1. `pipenv install`
2. Build [MyPaint](https://github.com/mypaint/mypaint) (See the following)
3. Append the path of MyPaint's build directory to `.env`:

```
PYTHONPATH=<path-to-my-paint>build/lib.macosx-10.13-x86_64-3.6:$PYTHONPATH
```

## Run a pre-trained model

`pipenv run python demo.py many trained_models/mnist/64296000 result.png`

## How to train

`pipenv run python train.py settings/default.yaml <directory-to-put-logs>`

Details of training options available on `settings/default.yaml`.

## How to run a trained model

`pipenv run python demo.py many <path-to-snapshot> many.png`

You can choose a demo mode from `static`, `many`, `movie`, and `json`:

An example of static demo:

![](images/static.png)

Many demo:

![](images/many.png)

Movie:

![](images/movie.gif)

## Setup this project by Docker

`cd docker`
`docker build ./ -t chainer_spiral`
`cd ../  # go to directory of this repo`

```
docker run -it -v `pwd`:/root -u`id -u`:`id -g` chainer_spiral bash
```

## How to install MyPaint for this project

### Install dependencies (CentOS)

```
$ sudo yum install gcc gobject-introspection-devel json-c-devel glib2-devel git python autoconf intltool gettext libtool swig python-setuptools gettext gcc-c++ python-devel numpy gtk3-devel pygobject3-devel libpng-devel lcms2-devel json-c-devel gtk3 gobject-introspection
```

### Install libmypaint

```
$ git clone https://github.com/mypaint/libmypaint
$ cd libmypaint
# git checkout 0c07191409bd257084d4ea7576deb832aac8868b
$ ./autogen.sh
$ ./configure --prefix=<your-installation-prefix>
$ make install
```

Make sure that `<your-installation-prefix>/lib` is in `LD_LIBRARY_PATH` and `PYTHONPATH`. Also `PKG_CONFIG_PATH` shoud include `<your-installation-prefix>/lib` and `<your-installation-prefix>/share`.

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

### Check your installation

Go to this repo's directory and run tests by `pipenv run test`
