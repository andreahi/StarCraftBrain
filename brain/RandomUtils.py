import numpy as np


def normalize(vals):
    return (vals - min(vals)) / (max(vals) - min(vals))


def weighted_random_index(weights):
    weights = get_probability(weights)
    return np.random.choice(range(len(weights)), p=weights)


def get_probability(weights):
    weights = weights / np.sum(weights)
    return weights
