# ChainerSPIRAL

Chainer implementation of Synthesizing Programs for Images using Reinforced Adversarial Learning (SPIRAL).

## Dependencies

- [Pipenv](https://pipenv.readthedocs.io/en/latest/)

## How to start

1. Clone this repo and `cd <path-to-repo>`
2. Run `echo PYTHONPATH=`pwd`/chainerrl > .env`
3. `pipenv install`

## How to run
- Run IMPALA example: `pipenv run impala`
- All the unit tests without GPU: `pipenv run cputest`
- All the unit tests with GPU: `pipenv run gputest`

