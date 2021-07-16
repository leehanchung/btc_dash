import numpy as np
import tensorflow as tf


def create_tfds_from_np(
    *,
    data: np.ndarray,
    window_size: int = 31,
    shift_size: int = 1,
    stride_size: int = 1,
    batch_size: int = 15,
) -> tf.data.Dataset:
    """Generates tf.data dataset using a given numpy array using tf.data API

    Args:
        data: np.ndarray data in numpy array, to be flattened
        window_size: size of the moving window
        shift_size: step size of the moving window,
            e.g. [0, 1, 2, 3, 4, 5, 6] with shift 2 and window 3
            -> [0, 1, 2], [2, 3, 4], ...
        stride_size: sampling size of the moving window,
            e.g., [0, 1, 2, 3, 4, 5, 6] with stride 2 and window 3
            -> [0, 2, 4], [1, 3, 5], ...
        batch_size: batch size of the created data

    Returns:
        tf.data.Dataset

    """
    data = tf.data.Dataset.from_tensor_slices(data.reshape(-1, 1))
    data = data.window(
        size=window_size,
        shift=shift_size,
        stride=stride_size,
        drop_remainder=True,
    )
    data = data.flat_map(
        lambda window: window.batch(window_size, drop_remainder=True)
    )
    data = data.map(lambda window: (window[:-1], tf.reshape(window[-1:], [])))
    data = data.shuffle(batch_size).batch(batch_size).cache().repeat()

    return data
