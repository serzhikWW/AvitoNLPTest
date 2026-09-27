"""Эксперимент 8: кривая обучения ранкера — упирается ли он в число обучающих запросов.

Финальная конфигурация (ft-val, фьюжн 0.5/1/0.1, --geo --noloc 100). 5-fold по запросам;
в каждом фолде ранкер учится на доле frac обучающих запросов, тест — всегда целый фолд.
Если recall продолжает расти к frac=1, расширение обучающей выборки ранкера оправдано.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import CACHE_DIR
from src.ranker import LGB_PARAMS, feats_tag, ranker_features
from src.validation import build_hard_corpus, load_split, recall_at_k

_, queries, qrels, _ = load_split()
ids = build_hard_corpus().item_id.values
X = pd.read_parquet(CACHE_DIR / f"ranker_feats_val_{feats_tag('models/multilingual-e5-base-ft-val', (0.5, 1.0, 0.1), True, 100)}.parquet")
feats = ranker_features(X)
loc_n = queries.set_index("query_id").search_location_id.map(
    pd.Series(build_hard_corpus().item_location_id).value_counts()).fillna(0)
rng = np.random.default_rng(0)

for frac in [0.25, 0.5, 0.75, 1.0]:
    oof = np.zeros(len(X))
    for tr_idx, te_idx in GroupKFold(5).split(X, groups=X.query_id):
        tr = X.iloc[tr_idx]
        qs = tr.query_id.unique()
        keep = set(rng.choice(qs, int(len(qs) * frac), replace=False))
        tr = tr[tr.query_id.isin(keep)].sort_values("query_id", kind="stable")
        m = lgb.LGBMRanker(**LGB_PARAMS).fit(tr[feats], tr.label, group=tr.groupby("query_id", sort=True).size().values)
        oof[te_idx] = m.predict(X.iloc[te_idx][feats])
    d = X[["query_id", "item_idx"]].assign(s=oof).sort_values(["query_id", "s"], ascending=[True, False]).groupby("query_id").head(50)
    r = recall_at_k({q: list(ids[g.item_idx.values]) for q, g in d.groupby("query_id")}, qrels)
    n = loc_n.reindex(r.index)
    print(f"train queries ≈ {int(2400 * frac):5d}  R@50={r.mean():.4f}  megacity={r[n > 10000].mean():.4f}  region={r[n == 0].mean():.4f}")
