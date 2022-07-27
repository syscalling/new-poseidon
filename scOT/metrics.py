import numpy as np


def lp_error(preds: np.ndarray, targets: np.ndarray, p=1):
    num_samples, num_channels, _, _ = preds.shape
    preds = preds.reshape(num_samples, num_channels, -1)
    targets = t