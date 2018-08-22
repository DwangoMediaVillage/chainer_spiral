# ChainerSPIRAL

Chainer implementation of Synthesizing Programs for Images using Reinforced Adversarial Learning (SPIRAL).

## Dependencies

- [Pipenv](https://pipenv.readthedocs.io/en/latest/)


## How to start

1. Clone this repo and `cd <path-to-repo>`, and `pipenv install`
2. Run `echo PYTHONPATH=`pwd`/chainerrl > .env`
3. Build mypaint: `https://github.com/mypaint/mypaint`, and run `python setup.py test` to produce a shared library file.
4. Append the path of mypaint directory to `.env`'s PYTHONPATH
5. If you want to test with MyPaintEnv, append path of `mypaint_env.py` to `PYTHONPATH`.

## How to run

Start a training example: `pipenv run spiral`