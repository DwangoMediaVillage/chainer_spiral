import numpy as np

def get_model_param_sum(model):
    sum = 0.0
    for p in model.params():
        sum += p.data.sum()
    return sum