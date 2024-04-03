from sklearn import datasets, utils
import numpy as np


# --------------------------------------------------------------------------------------#
# DATA
# --------------------------------------------------------------------------------------#
def digit_data() -> utils.Bunch:
    return datasets.load_digits()


def target(digit_data: utils.Bunch) -> np.ndarray:
    return digit_data.target


def target_names(digit_data: utils.Bunch) -> np.ndarray:
    return digit_data.target_names


def feature_matrix(digit_data: utils.Bunch) -> np.ndarray:
    return digit_data.data
