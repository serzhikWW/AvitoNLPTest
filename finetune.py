"""Дообучение bi-encoder'а (E5) на кликах из train: контрастивное обучение.

Данные: пары (запрос, выбранное по нему объявление). Лосс — MultipleNegativesRanking:
для каждого запроса в батче его объявление — позитив, объявления остальных запросов
батча — негативы (in-batch negatives). Cached-версия лосса позволяет держать большой
батч (много негативов) при ограниченной видеопамяти: градиенты считаются по мини-батчам.

Детали, важные для этой задачи:
  * бенчмарк — выборка УНИКАЛЬНЫХ запросов, а в train головные запросы («массаж»)
    повторяются тысячи раз. Ограничиваем число пар на один текст запроса (MAX_PER_QUERY),
    иначе модель переобучится на голову распределения;
  * NO_DUPLICATES-сэмплер не кладёт в один батч одинаковые тексты: иначе позитив
    одного примера оказался бы «негативом» для такого же запроса.

Режимы:
  --mode val   обучение на train-fit (без val-запросов) -> честная оценка на val v2;
  --mode full  обучение на всём train -> модель для финального answer.csv.

Запуск (на GPU):
  python finetune.py --mode val  --model intfloat/multilingual-e5-base
  python finetune.py --mode full --model intfloat/multilingual-e5-base
Результат: models/<model>-ft-<mode>/
"""
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from src.data import ROOT, load_train
from src.dense import device, item_text
from src.validation import load_split

MAX_PER_QUERY = 20
SEED = 42


def build_pairs(train: pd.DataFrame) -> Dataset:
    pairs = train.drop_duplicates(["search_query", "item_id"])
    # не больше MAX_PER_QUERY пар на один текст запроса (случайно, но детерминированно)
    pairs = pairs.sample(frac=1.0, random_state=SEED).groupby("search_query").head(MAX_PER_QUERY)
    anchors = ("query: " + pairs.search_query).tolist()
    positives = ["passage: " + t for t in item_text(pairs)]
    print(f"pairs: {len(anchors)} ({pairs.search_query.nunique()} unique queries)")
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "full"], required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256, help="логический батч = число in-batch негативов + 1")
    ap.add_argument("--mini_batch", type=int, default=64, help="уменьшить при нехватке видеопамяти")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_pairs", type=int, default=0, help="для отладки: обрезать данные")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train = load_split()[0] if args.mode == "val" else load_train()
    data = build_pairs(train)
    if args.max_pairs:
        data = data.select(range(args.max_pairs))

    dev = device()
    # на Apple MPS SDPA-attention не поддерживает dropout при обучении -> eager; на CUDA оставляем SDPA
    mkw = {"attn_implementation": "eager"} if dev == "mps" else {}
    model = SentenceTransformer(args.model, device=dev, model_kwargs=mkw)
    model.max_seq_length = 128
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    suffix = "-debug" if args.max_pairs else ""
    out = ROOT / "models" / f"{args.model.split('/')[-1]}-ft-{args.mode}{suffix}"
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy="no",
        seed=SEED,
        report_to="none",
    )
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=data, loss=loss).train()
    model.save(str(out))
    print("saved to", out)


if __name__ == "__main__":
    main()
