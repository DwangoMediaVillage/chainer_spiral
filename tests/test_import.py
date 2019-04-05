from nose.tools import ok_, eq_

def test_import_paintlib():
    # try to import mypaintlib
    from lib import mypaintlib

def test_import_gym():
    # try to import gym
    import gym

def test_import_chainerrl():
    # try to import chainerrl
    import chainerrl

def test_import_spiral():
    # try to import spiral env
    from chainer_spiral.environments import MyPaintEnv

    # try to import spiral agent
    from chainer_spiral.agents.spiral import SPIRAL

    # try to import spiral models
    from chainer_spiral.models.spiral import SpiralToyModel, SpiralToyDiscriminator
    from chainer_spiral.models.spiral import SpiralMnistModel, SpiralMnistDiscriminator
