"""Проверка формата answer.csv по требованиям ТЗ перед отправкой."""
import re
import sys

import pandas as pd

from src.data import load_bench_items, load_bench_queries

path = sys.argv[1] if len(sys.argv) > 1 else "answer.csv"
a = pd.read_csv(path, dtype=str, keep_default_na=False)
q = load_bench_queries()
valid = set(load_bench_items(columns=["item_id"]).item_id)

assert list(a.columns) == ["query_id", "answer"], a.columns
assert a.query_id.is_unique and set(a.query_id) == set(q.query_id), "query_id mismatch"
assert a.query_id.str.len().eq(16).all()
for s in a.answer:
    ids = s.split(" ")
    assert 0 < len(ids) <= 50 and len(set(ids)) == len(ids), "size/dup"
    assert all(re.fullmatch(r"[0-9a-f]{16}", i) and i in valid for i in ids), "bad item_id"
print(f"OK: {len(a)} rows, mean {a.answer.str.split().str.len().mean():.1f} items/row")
