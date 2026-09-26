"""Общие утилиты ретривала: батчевый top-k по плотным скорам."""
import numpy as np


def topk_indices(scores: np.ndarray, k: int = 50) -> np.ndarray:
    """Индексы top-k по каждой строке (в порядке убывания скора)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)


def batched(n: int, size: int = 256):
    for s in range(0, n, size):
        yield slice(s, min(s + size, n))
