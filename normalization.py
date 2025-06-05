"""Different normalizations"""
import numpy as np


def normalization_01(x):
    """To normalize the feature x within range [0,1]

    Args:
        x (array): feature

    Returns:
        array: Normalized feature
    """

    normalized_x = [round(((x_i) - min(x)) / (max(x) - min(x)), 4) for x_i in x]

    return normalized_x


def normalization_11(x):
    """To normalize the feature x within range [-1,1]

    Args:
        x (array): feature

    Returns:
        array: Normalized feature
    """

    normalized_x = [
        round((2 * (x_i) - min(x) - max(x)) / (max(x) - min(x)), 4) for x_i in x
    ]

    return normalized_x


def mean_normalization(x):
    """To normalize the feature x within range [-1,1]

    Args:
        x (array): feature

    Returns:
        array: Normalized feature
    """

    normalized_x = [round(((x_i) - np.mean(x)) / (max(x) - min(x)), 4) for x_i in x]

    return normalized_x


x = [100, 120, 200, 4563, 23, 56, 788, 1000, 3400, 40, 400, 2500, 4000, 1700]

print(f"Normalized x in range[0,1] :  {normalization_01(x)}")
print(f"Normalized x in range[0,1] : {normalization_11(x)}")
print(f"Normalized x in range[0,1] : {mean_normalization(x)}")
