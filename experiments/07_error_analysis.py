"""Эксперимент 7: анализ ошибок финального ранкера на val v2.

Берём OOF-предсказания (5-fold по запросам) финальной конфигурации:
дообученная e5-base (ft-val), фьюжн 0.5/1/0.1, ранкер без статичных признаков объявления.
Для каждого пропущенного релевантного объявления определяем причину:
  * not_candidate — не попало в кандидаты (top-300 BM25 / dense / фьюжн);
  * ranked_out    — было среди кандидатов, но ранкер поставил его ниже 50-го места.
Для ranked_out смотрим, каких сигналов не хватило (локация, пересечение слов, подкатегория).
Затем — recall по сегментам запросов и примеры промахов.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import CACHE_DIR
from src.ranker import LGB_PARAMS, feats_tag, ranker_features
from src.text import normalize
from src.validation import build_hard_corpus, load_split, recall_at_k

MODEL, FUSE = "models/multilingual-e5-base-ft-val", (0.5, 1.0, 0.1)
pd.set_option("display.width", 200, "display.max_colwidth", 60)

train_fit, queries, qrels, _ = load_split()
corpus = build_hard_corpus()
X = pd.read_parquet(CACHE_DIR / f"ranker_feats_val_{feats_tag(MODEL, FUSE)}.parquet")
feats = ranker_features(X)

# --- OOF-скоры ранкера
oof = np.zeros(len(X))
for tr_idx, te_idx in GroupKFold(5).split(X, groups=X.query_id):
    tr = X.iloc[tr_idx].sort_values("query_id", kind="stable")
    m = lgb.LGBMRanker(**LGB_PARAMS).fit(tr[feats], tr.label, group=tr.groupby("query_id", sort=True).size().values)
    oof[te_idx] = m.predict(X.iloc[te_idx][feats])
X["score"] = oof
X["rank"] = X.groupby("query_id").score.rank(ascending=False, method="first")
ids = corpus.item_id.values
X["item_id"] = ids[X.item_idx.values]
top = X[X["rank"] <= 50]
preds = {q: list(g.item_id) for q, g in top.groupby("query_id")}
r = recall_at_k(preds, qrels)
print(f"OOF Recall@50 = {r.mean():.4f}\n")

# --- причины промахов по каждому релевантному объявлению
q = queries.set_index("query_id")
rel = qrels.merge(X[["query_id", "item_id", "rank", "loc_logp", "same_loc", "title_cover", "bm25",
                     "cos_rank", "bm25_rank", "p_mc_lemma", "mc_share_top50"]], how="left")
rel["found"] = rel["rank"] <= 50
rel["reason"] = np.where(rel.found, "found", np.where(rel["rank"].isna(), "not_candidate", "ranked_out"))
print("Судьба релевантных объявлений:")
print(rel.reason.value_counts(normalize=True).round(3).to_string(), "\n")

ro = rel[rel.reason == "ranked_out"]
fd = rel[rel.reason == "found"]
print("ranked_out vs found (средние признаки релевантного объявления):")
cmp = pd.DataFrame({
    "found": fd[["same_loc", "title_cover", "bm25", "cos_rank", "bm25_rank", "p_mc_lemma", "mc_share_top50"]].mean(),
    "ranked_out": ro[["same_loc", "title_cover", "bm25", "cos_rank", "bm25_rank", "p_mc_lemma", "mc_share_top50"]].mean(),
})
print(cmp.round(3).to_string(), "\n")
print("ranked_out: место, на которое ранкер поставил позитив:")
print(pd.cut(ro["rank"], [50, 75, 100, 200, 1000]).value_counts(normalize=True).sort_index().round(3).to_string(), "\n")

# --- recall по сегментам запросов
corp_locs = corpus.item_location_id.value_counts()
seg = pd.DataFrame({"recall": r})
qq = q.loc[seg.index]
seg["seen"] = qq.seen.values
seg["region_query"] = ~qq.search_location_id.isin(corp_locs.index).values
seg["q_words"] = qq.search_query.str.split().str.len().clip(upper=5).values
seg["has_params"] = (qq.search_infm_params_text != "").values
seg["n_cand"] = X.groupby("query_id").size().reindex(seg.index).values
loc_items = qq.search_location_id.map(corp_locs).fillna(0).values
seg["city_size"] = pd.cut(loc_items, [-1, 0, 500, 3000, 10000, 1e6], labels=["region", "<500", "500-3k", "3k-10k", ">10k"])
for col in ["seen", "region_query", "q_words", "has_params", "city_size"]:
    t = seg.groupby(col, observed=True).recall.agg(["mean", "size"])
    t["share_of_loss"] = seg.assign(loss=1 - seg.recall).groupby(col, observed=True).loss.sum() / (1 - seg.recall).sum()
    print(f"-- по {col}:")
    print(t.round(3).to_string(), "\n")

# --- примеры промахов
cols = corpus.set_index("item_id")
def title(i):
    return cols.item_title_raw.get(i, "?")
miss = rel[~rel.found].sample(12, random_state=0)
print("Примеры промахов (запрос | релевантное | причина, место | top-1 ранкера):")
for _, row in miss.iterrows():
    t1 = top[top.query_id == row.query_id].sort_values("rank").item_id.head(1)
    print(f"  «{q.loc[row.query_id, 'search_query']}» | {title(row.item_id)[:50]} | {row.reason}, "
          f"{'—' if pd.isna(row['rank']) else int(row['rank'])} | {title(t1.iloc[0])[:45] if len(t1) else '-'}")

# --- пересечение слов запроса и релевантного объявления (для not_candidate)
nc = rel[rel.reason == "not_candidate"].copy()
tok = corpus.set_index("item_id")
nc["overlap"] = [
    len(set(normalize(q.loc[a, "search_query"])) & set(normalize(tok.item_title_raw.get(b, "") + " " + tok.item_description_raw.get(b, ""))))
    for a, b in zip(nc.query_id, nc.item_id)
]
print("\nnot_candidate: сколько слов запроса есть в тексте релевантного:", nc.overlap.value_counts().sort_index().to_dict())
nc_loc = [corpus.set_index("item_id").item_location_id.get(b) == q.loc[a, "search_location_id"] for a, b in zip(nc.query_id, nc.item_id)]
print("not_candidate: доля с совпадающей локацией:", round(float(np.mean(nc_loc)), 3))

# --- гипотеза: локация не совпадает, потому что исполнитель работает удалённо (отвергнута)
remote_re = r"Удал[её]нно|[Оо]нлайн|По всей России"
city = train_fit.search_location_id.isin(set(corpus.item_location_id))
t = train_fit[city].assign(remote=lambda d: d.item_infm_params_text.str.contains(remote_re, regex=True))
mism = (t.search_location_id != t.item_location_id)
print(f"\ntrain, поиск по городу: P(другая локация | удалённое объявление) = {mism[t.remote].mean():.3f}, "
      f"P(другая локация | обычное) = {mism[~t.remote].mean():.3f}")

# --- запросы по региону: насколько «маловероятна» по приору локация релевантного объявления
from src.location import LocationPrior
lp = LocationPrior().fit(train_fit)
rq = rel.merge(corpus[["item_id", "item_location_id"]], how="left")
rq["search_loc"] = q.loc[rq.query_id, "search_location_id"].values
rq = rq[~rq.search_loc.isin(corp_locs.index)]
pr = np.array([lp.probs(s).get(l, 0) for s, l in zip(rq.search_loc, rq.item_location_id)])
print("запросы по региону: квантили P(item_loc | search_loc) у релевантных (10/25/50%):",
      np.round(np.quantile(pr, [.1, .25, .5]), 4), " найдено:", round(rq.found.mean(), 3))
