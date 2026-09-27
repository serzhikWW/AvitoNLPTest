"""Эксперимент 10: второй уровень ранжирования со скором cross-encoder'а (val v2).

Вход: cache/stage1_val.parquet (top-100 первого уровня с OOF-скорами, export_ce_pairs.py)
      и cache/ce_scores_val.parquet (скоры модели ce-val, score_cross_encoder.py).
Второй уровень — LightGBM lambdarank на top-100: признаки первого уровня + его скор и ранг
+ скор cross-encoder'а и его ранг внутри запроса. Качество — 5-fold по запросам.
Сравниваем с первым уровнем (top-50 по OOF-скору) и с вариантом без cross-encoder'а
(чтобы отделить эффект скора CE от эффекта «второго прохода» ранкера).
"""
import sys

import pandas as pd

from src.data import CACHE_DIR
from src.ranker import oof_scores, ranker_features
from src.validation import build_hard_corpus, load_split, recall_at_k

scores_file = sys.argv[1] if len(sys.argv) > 1 else "ce_scores_val.parquet"
_, queries, qrels, _ = load_split()
loc_n = queries.set_index("query_id").search_location_id.map(build_hard_corpus().item_location_id.value_counts()).fillna(0)

S = pd.read_parquet(CACHE_DIR / "stage1_val.parquet").merge(pd.read_parquet(CACHE_DIR / scores_file), on=["query_id", "item_id"])
S["ce_rank"] = S.groupby("query_id").ce_score.rank(ascending=False, method="first")
S["ce_gap"] = S.ce_score - S.groupby("query_id").ce_score.transform("max")


def show(name, score):
    top = S.assign(s=score).sort_values(["query_id", "s"], ascending=[True, False]).groupby("query_id").head(50)
    r = recall_at_k({q: list(g.item_id) for q, g in top.groupby("query_id")}, qrels)
    n = loc_n.reindex(r.index)
    print(f"{name:<34s} R@50={r.mean():.4f}  region={r[n == 0].mean():.4f}  megacity={r[n > 10000].mean():.4f}")


base = ranker_features(S.drop(columns=["stage1", "stage1_rank", "ce_score", "ce_rank", "ce_gap", "item_id"]))
show("stage1 (ранкер, отправка 4)", S.stage1.values)
show("cross-encoder alone", S.ce_score.values)
show("stage2 без CE", oof_scores(S, base + ["stage1", "stage1_rank"]))
show("stage2 + CE", oof_scores(S, base + ["stage1", "stage1_rank", "ce_score", "ce_rank", "ce_gap"]))
