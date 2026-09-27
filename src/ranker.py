"""Кандидаты + признаки для LightGBM-ранкера (второй уровень кандидатогенерации).

Зачем: на val v2 объединение top-300 BM25 и top-300 эмбеддингов содержит
~94% релевантных, а в top-50 фьюжна попадает ~85%. Нужное объявление обычно
уже среди кандидатов; нужно лучше выбрать 50 из нескольких сотен похожих.

Кандидаты запроса = top-N по трём скорам (BM25+loc, dense+loc, линейный фьюжн).
Признаки пары (запрос, объявление):
  * текстовые: BM25 по всему тексту и отдельно по заголовку, косинус эмбеддингов,
    ранги по ним, нормированный BM25, доля слов запроса в заголовке;
  * локация: log P(item_loc | search_loc), флаг совпадения локации, «регион ли» поиск;
  * подкатегория: P(microcat | текст запроса) и P(microcat | леммы запроса) по train,
    доля этой подкатегории среди top-50 фьюжна (псевдорелевантная обратная связь);
  * свойства объявления: цена, рейтинг, отзывы, флаги, длина текста.

Сознательно НЕ используются признаки истории объявления (сколько раз его кликали):
в val v2 история есть у 49% корпуса (дистракторы берутся из train), в бенчмарке —
у 10%. Ранкер выучил бы на val связь, которой в бенчмарке нет.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .bm25 import SparseBM25
from .location import GeoPrior, LocationPrior
from .retrieval import batched, topk_indices
from .text import normalize
from .tokens import item_tokens

FUSE = (0.5, 1.0, 0.05)  # a*bm25_norm + b*cos + w*logP(loc), лучший фьюжн на val v2

# Статичные признаки объявления (цена, отзывы, длина текста...) в финальный ранкер не идут:
# в val позитивы и дистракторы взяты из train, остальной корпус — из бенчмарка, и по этим
# признакам источник слегка угадывается (AUC 0.58, в основном за счёт числа отзывов).
# Абляция: без них Recall@50 на val 0.8826 против 0.8844 — почти без потерь, зато
# ранкер не может выучить артефакт «объявление из бенчмарка -> нерелевантно».
STATIC_ITEM = ["log_price", "rating", "log_reviews", "phone_hidden", "msg_forbidden", "title_len", "log_desc_len"]
LGB_PARAMS = dict(
    objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, lambdarank_truncation_level=60,
    random_state=42, deterministic=True, force_row_wise=True, verbose=-1,
)


def feats_tag(model_name: str, fuse: tuple = FUSE, geo: bool = False, n_noloc: int = 0) -> str:
    """Ключ кэша признаков val: модель эмбеддингов + настройки, отличные от умолчаний."""
    tag = model_name.rstrip("/").split("/")[-1]
    tag += "" if tuple(fuse) == FUSE else "_fuse" + "_".join(map(str, fuse))
    tag += "_geo" if geo else ""
    tag += f"_noloc{n_noloc}" if n_noloc else ""
    return tag


def ranker_features(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ("query_id", "item_idx", "label", *STATIC_ITEM)]


class MicrocatStats:
    """Распределения подкатегорий по запросу, выученные из train."""

    def __init__(self, min_count: int = 5):
        self.min_count = min_count

    def fit(self, train: pd.DataFrame):
        pairs = train.drop_duplicates(["search_query", "item_id"])
        g = pairs.groupby(["search_query", "item_microcat_id"]).size()
        tot = g.groupby(level=0).transform("sum")
        self.by_text = defaultdict(dict)  # текст -> {microcat: P}
        for (t, mc), p in (g / tot).items():
            self.by_text[t][mc] = p
        lemma_mc = defaultdict(Counter)
        for q, mc in zip(pairs.search_query.values, pairs.item_microcat_id.values):
            for lem in set(normalize(q)):
                lemma_mc[lem][mc] += 1
        self.by_lemma = {}
        for lem, cnt in lemma_mc.items():
            n = sum(cnt.values())
            if n >= self.min_count:
                self.by_lemma[lem] = {mc: c / n for mc, c in cnt.items()}
        return self

    def lemma_probs(self, lemmas: list[str]) -> dict:
        """Средняя по леммам запроса P(microcat | лемма); леммы без статистики пропускаются."""
        dists = [self.by_lemma[l] for l in set(lemmas) if l in self.by_lemma]
        if not dists:
            return {}
        out = Counter()
        for d in dists:
            out.update(d)
        return {k: v / len(dists) for k, v in out.items()}


class FeatureBuilder:
    def __init__(self, corpus: pd.DataFrame, train: pd.DataFrame, item_emb: np.ndarray, query_encoder,
                 n_cand: int = 300, fuse: tuple = FUSE, geo: bool = False,
                 n_noloc: int = 0, w_noloc: float = 0.01):
        """geo / n_noloc — доработки по итогам анализа ошибок (experiments/07_error_analysis.py):
          geo      гео-признаки: расстояние до «центра» локации поиска, нормированное на её радиус,
                   встречалась ли пара (локация поиска, локация объявления) в train, ранг локации
                   объявления в распределении P(item_loc | search_loc);
          n_noloc  доп. источник кандидатов: top-n_noloc фьюжна с маленьким весом локации w_noloc.
                   87% релевантных, не попавших в кандидаты, — из «чужой» локации.
        По умолчанию выключены, чтобы воспроизводились отправки 2 и 3.
        """
        self.corpus = corpus.reset_index(drop=True)
        self.n_cand = n_cand
        self.fuse = fuse
        self.geo, self.n_noloc, self.w_noloc = geo, n_noloc, w_noloc
        tok = item_tokens(self.corpus, "lemma")
        self.bm = SparseBM25().fit_fields([tok.title.values, tok.params.values, tok.desc.values], [3.0, 1.0, 1.0])
        self.bm_title = SparseBM25().fit(tok.title.values)
        self.title_sets = [set(t.split()) for t in tok.title]
        self.E = item_emb.astype(np.float32)
        self.encode = query_encoder
        self.lp = LocationPrior().fit(train)
        self.mc = MicrocatStats().fit(train)
        self.train_texts = set(train.search_query)
        c = self.corpus
        self.item_locs = c.item_location_id.values
        self.loc_set = set(self.item_locs)
        self.item_mc = c.item_microcat_id.values
        if geo:
            self.gp = GeoPrior().fit(train, c)
            self.item_lat = c.item_latitude.astype(float).values
            self.item_lon = c.item_longitude.astype(float).values
        # статичные признаки объявления
        self.item_feats = pd.DataFrame({
            "log_price": np.log1p(c.item_price.fillna(0).clip(lower=0)),
            "rating": c.item_rating.fillna(0),
            "log_reviews": np.log1p(c.item_rating_reviews_count.fillna(0)),
            "phone_hidden": c.item_is_phone_hidden.astype(float),
            "msg_forbidden": c.item_is_message_forbidden.astype(float),
            "title_len": c.item_title_raw.str.len(),
            "log_desc_len": np.log1p(c.item_description_raw.str.len()),
        })

    def build(self, queries: pd.DataFrame, qrels: pd.DataFrame | None = None) -> pd.DataFrame:
        """Строки (query_id, item_idx, признаки..., [label])."""
        q = queries.reset_index(drop=True)
        qlem = [normalize(t) for t in q.search_query]
        Qe = self.encode(q.search_query.tolist()).astype(np.float32)
        priors = {l: self.lp.log_prior(l, self.item_locs) for l in q.search_location_id.unique()}
        rel = qrels.groupby("query_id").item_id.apply(set).to_dict() if qrels is not None else {}
        ids = self.corpus.item_id.values
        a, b, w = self.fuse
        parts = []
        for sl in batched(len(q), 128):
            B = self.bm.scores(qlem[sl])
            Bt = self.bm_title.scores(qlem[sl])
            C = Qe[sl] @ self.E.T
            L = np.vstack([priors[l] for l in q.search_location_id.values[sl]])
            Bn = B / np.maximum(B.max(1, keepdims=True), 1e-6)
            F = a * Bn + b * C + w * L
            cand_sets = [topk_indices(B + 2.0 * L, self.n_cand),
                         topk_indices(C + 0.1 * L, self.n_cand),
                         topk_indices(F, self.n_cand)]
            top50_fuse = cand_sets[2][:, :50]
            if self.n_noloc:
                cand_sets.append(topk_indices(a * Bn + b * C + self.w_noloc * L, self.n_noloc))
            for j, i in enumerate(range(sl.start, sl.stop)):
                cand = np.unique(np.concatenate([cs[j] for cs in cand_sets]))
                row = q.iloc[i]
                terms = set(qlem[i])
                mc_text = self.mc.by_text.get(row.search_query, {})
                mc_lem = self.mc.lemma_probs(qlem[i])
                mc_top = Counter(self.item_mc[top50_fuse[j]])
                cmc = self.item_mc[cand]
                d = {
                    "query_id": row.query_id,
                    "item_idx": cand,
                    "bm25": B[j, cand], "bm25_norm": Bn[j, cand], "bm25_title": Bt[j, cand],
                    "cos": C[j, cand], "loc_logp": L[j, cand], "fuse": F[j, cand],
                    "same_loc": (self.item_locs[cand] == row.search_location_id).astype(float),
                    "title_cover": [len(terms & self.title_sets[k]) / max(len(terms), 1) for k in cand],
                    "p_mc_text": [mc_text.get(m, 0.0) for m in cmc],
                    "p_mc_lemma": [mc_lem.get(m, 0.0) for m in cmc],
                    "mc_share_top50": [mc_top.get(m, 0) / 50 for m in cmc],
                    "q_len": len(qlem[i]),
                    "q_seen": float(row.search_query in self.train_texts),
                    "q_region": float(row.search_location_id not in self.loc_set),
                    "q_has_params": float(bool(row.search_infm_params_text)),
                }
                if self.geo:
                    s_loc = row.search_location_id
                    dist, dist_norm = self.gp.features(s_loc, self.item_lat[cand], self.item_lon[cand])
                    obs = self.lp.dist.get(s_loc, {})
                    loc_rank = {l: r for r, (l, _) in enumerate(sorted(obs.items(), key=lambda kv: -kv[1]), 1)}
                    clocs = self.item_locs[cand]
                    d.update({
                        "geo_log_dist": np.log1p(dist),
                        "geo_dist_norm": np.log1p(dist_norm),
                        "loc_seen": np.array([l in obs for l in clocs], dtype=float),
                        "loc_rank": np.log1p([loc_rank.get(l, 1000) for l in clocs]),
                    })
                df = pd.DataFrame(d)
                if qrels is not None:
                    df["label"] = np.isin(ids[cand], list(rel.get(row.query_id, ()))).astype(int)
                parts.append(df)
        out = pd.concat(parts, ignore_index=True)
        # ранги внутри кандидатов запроса (устойчивее к масштабу скоров, чем сами скоры)
        for f in ["bm25", "cos", "fuse", "bm25_title"]:
            out[f"{f}_rank"] = out.groupby("query_id")[f].rank(ascending=False, method="first")
        return pd.concat([out, self.item_feats.iloc[out.item_idx.values].reset_index(drop=True)], axis=1)
