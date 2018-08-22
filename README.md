# ChainerSPIRAL

Chainer implementation of Synthesizing Programs for Images using Reinforced Adversarial Learning (SPIRAL).

## Dependencies

- [Pipenv](https://pipenv.readthedocs.io/en/latest/)
- [mypaint](https://github.com/mypaint/mypaint)

## How to start

1. Clone this repo and `cd <path-to-repo>`, and `pipenv install`
2. Build mypaint: `https://github.com/mypaint/mypaint`, and run `python setup.py test` to produce a shared library file.
3. Append the path of mypaint directory to `.env`'s PYTHONPATH
4. Append the path of a brush file (`myb`) to `.env`'s BRUSHINFO (e.g. `BRUSHINFO=settings/my_simple_brush.myb`)

## How to run

Start a training example: `pipenv run train`