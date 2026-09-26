"""Локационный приор P(локация объявления | локация поиска).

EDA: если локация поиска — город, в 93% случаев выбранное объявление из этого же
города. Если это регион/страна (напр. 107620 — Московская обл., 621540 — вся РФ),
в корпусе объявлений с таким location_id нет вовсе, а выбранные объявления лежат
в «дочерних» городах (для 107620 — в основном Москва 637640).

Распределение оцениваем по train-fit (счётчики пар search_loc -> item_loc)
со сглаживанием: к наблюдаемым парам добавляется псевдосчётчик для «своей»
локации, чтобы редкие локации без истории всё равно тянули к себе свои объявления.
"""
import numpy as np
import pandas as pd


class LocationPrior:
    def __init__(self, self_pseudo: float = 5.0, floor: float = 1e-4):
        self.self_pseudo = self_pseudo
        self.floor = floor  # вероятность «любой другой» локации

    def fit(self, train: pd.DataFrame):
        pairs = train.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        self.pairs = pairs
        self.dist = {s: dict(zip(g.item_location_id, g.n)) for s, g in pairs.groupby("search_location_id")}
        return self

    def probs(self, search_loc: int) -> dict:
        d = dict(self.dist.get(search_loc, {}))
        d[search_loc] = d.get(search_loc, 0) + self.self_pseudo
        tot = sum(d.values())
        return {k: v / tot for k, v in d.items()}

    def log_prior(self, search_loc: int, item_locs: np.ndarray) -> np.ndarray:
        """log P(item_loc | search_loc) для каждого объявления корпуса."""
        p = self.probs(search_loc)
        vals = pd.Series(item_locs).map(p).fillna(0).to_numpy()
        return np.log(np.maximum(vals, self.floor)).astype(np.float32)
