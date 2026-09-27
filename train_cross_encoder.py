"""Дообучение cross-encoder'а на кликах из train (запускать на GPU).

Cross-encoder читает запрос и объявление ОДНОЙ последовательностью
«[запрос] [SEP] [объявление]», и attention работает между всеми их токенами.
Поэтому он различает тонкие случаи, которые bi-encoder (отдельные векторы запроса
и объявления) смешивает: «тонировка стёкол в квартире» — плёнка на окна, а не автотонировка.
Цена — прогон модели на каждую пару, поэтому он только переранжирует top-100 ранкера.

Данные: (запрос, объявление, метка) с бинарной кросс-энтропией.
  * позитив — объявление, выбранное по запросу (не больше MAX_PER_QUERY на текст запроса,
    чтобы головные запросы вроде «массаж» не забили выборку);
  * негативы (N_NEG на позитив) — «похожие объявления», главная причина промахов
    (experiments/09_megacity.py): объявления той же подкатегории и из той же локации,
    выбранные по ДРУГИМ запросам. Запрос-источник негатива не должен пересекаться с нашим
    по леммам больше чем на MAX_OVERLAP: иначе это та же потребность другими словами
    («откачка септика» vs «откачка септиков и выгребных ям») и негатив ложный — выбор между
    равнозначными исполнителями по тексту не выучить. Один негатив из той же подкатегории
    в любой локации — чтобы модель не выучила «другой город = нерелевантно»
    (локацию учитывает ранкер отдельно).

Режимы, как у bi-encoder'а:
  --mode val   на train-fit (без val-запросов) -> скор для обучения ранкера на val;
  --mode full  на всём train -> скор для бенчмарка.

Запуск:
  python train_cross_encoder.py --mode val
  python train_cross_encoder.py --mode full
Результат: models/<model>-ce-<mode>/ (по умолчанию models/cross-encoder-russian-msmarco-ce-<mode>/)
"""
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

from src.data import ROOT, load_train
from src.dense import device, item_text
from src.text import normalize
from src.validation import load_split

SEED = 42
MAX_PER_QUERY = 5
N_NEG = 3  # 2 из той же (локация, подкатегория) + 1 из той же подкатегории где угодно
MAX_OVERLAP = 0.5  # |леммы q ∩ леммы q'| / min(|q|, |q'|)


def build_examples(train: pd.DataFrame, max_pos: int, rng: np.random.Generator) -> Dataset:
    train = train.assign(text=item_text(train))
    pos = (train.drop_duplicates(["search_query", "item_id"])
           .sample(frac=1.0, random_state=SEED)
           .groupby("search_query").head(MAX_PER_QUERY)
           .head(max_pos))
    # пулы для негативов: (объявление, текст объявления, текст запроса, по которому его выбрали)
    items = train.drop_duplicates(["item_id", "search_query"])[
        ["item_id", "text", "search_query", "item_location_id", "item_microcat_id"]]
    by_loc_mc = items.groupby(["item_location_id", "item_microcat_id"]).indices
    by_mc = items.groupby("item_microcat_id").indices
    it_q = items.search_query.values
    it_id = items.item_id.values
    it_text = items.text.values
    lem = {t: set(normalize(t)) for t in set(it_q)}

    def same_need(q1, q2):
        a, b = lem[q1], lem[q2]
        return not a or not b or len(a & b) / min(len(a), len(b)) > MAX_OVERLAP

    def sample(pool, q, item_id, k):
        out = []
        if pool is None or len(pool) == 0:
            return out
        for j in rng.choice(pool, min(len(pool), 8 * k), replace=False):
            if it_id[j] != item_id and not same_need(q, it_q[j]):
                out.append(it_text[j])
                if len(out) == k:
                    break
        return out

    queries, docs, labels = [], [], []
    for q, iid, text, loc, mc in zip(pos.search_query.values, pos.item_id.values, pos.text.values,
                                     pos.item_location_id.values, pos.item_microcat_id.values):
        negs = sample(by_loc_mc.get((loc, mc)), q, iid, N_NEG - 1)
        negs += sample(by_mc.get(mc), q, iid, N_NEG - len(negs))
        for d, y in [(text, 1.0)] + [(n, 0.0) for n in negs]:
            queries.append(q)
            docs.append(d)
            labels.append(y)
    print(f"positives: {len(pos)}, examples: {len(labels)} (pos share {np.mean(labels):.2f})")
    return Dataset.from_dict({"query": queries, "doc": docs, "label": labels})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "full"], required=True)
    # RuBERT-base (178M), уже обучен как реранкер на русском MS MARCO. Более крупный
    # BAAI/bge-reranker-v2-m3 (568M) при полном дообучении с Adam требует ~9 ГБ и не влезает в 8 ГБ.
    ap.add_argument("--model", default="DiTy/cross-encoder-russian-msmarco",
                    help="быстрая альтернатива: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    ap.add_argument("--max_pos", type=int, default=100_000, help="число позитивов (x4 примеров)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--max_steps", type=int, default=-1, help="для отладки")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    train = load_split()[0] if args.mode == "val" else load_train()
    data = build_examples(train, args.max_pos, rng).shuffle(seed=SEED)

    dev = device()
    mkw = {"attn_implementation": "eager"} if dev == "mps" else {}  # см. finetune.py
    model = CrossEncoder(args.model, num_labels=1, max_length=args.max_length, device=dev, model_kwargs=mkw)
    # pos_weight: на 1 позитив N_NEG негативов — уравниваем вклад классов
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(float(N_NEG)))
    suffix = "-debug" if args.max_steps > 0 else ""
    out = ROOT / "models" / f"{args.model.split('/')[-1]}-ce-{args.mode}{suffix}"
    targs = CrossEncoderTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_strategy="no",
        seed=SEED,
        report_to="none",
    )
    CrossEncoderTrainer(model=model, args=targs, train_dataset=data, loss=loss).train()
    model.save(str(out))
    print("saved to", out)


if __name__ == "__main__":
    main()
