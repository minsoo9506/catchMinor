import numpy as np

from catchMinor.utils.debug import get_logger


def normal_only_train_split_tabular(
    X: np.ndarray, y: np.ndarray, train_ratio_in_normal: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """split data into 2 part, normal only train data and mix test data

    Args:
        X (np.ndarray): features
        y (np.ndarray): taget
        train_ratio_in_normal (float): train dat ratio in normal data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: normal_X_train, mix_X_test, normal_y_train, mix_y_test
    """
    logger = get_logger("train_test_split_tabular", "INFO")
    normal, abnormal = y == 0, y == 1
    normal_X, normal_y = X[normal], y[normal]
    abnormal_X, abnormal_y = X[abnormal], y[abnormal]
    num_train = int(len(normal_X) * train_ratio_in_normal)

    normal_X_train, normal_y_train = (
        normal_X[:num_train, :],
        normal_y[
            :num_train,
        ],
    )

    mix_X_test = np.concatenate((normal_X[num_train:, :], abnormal_X), axis=0)
    mix_y_test = np.concatenate((normal_y[num_train:], abnormal_y), axis=0)

    logger.info(f"num of (normal only) train data instance = {num_train}")
    logger.info(f"num of (mixed) test data instance = {len(mix_X_test)}")

    return normal_X_train, mix_X_test, normal_y_train, mix_y_test
