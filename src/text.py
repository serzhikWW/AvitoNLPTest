"""Нормализация русского текста для лексического поиска.

Три режима:
  * "raw"   — lowercase + ё->е + токенизация по буквам/цифрам;
  * "stem"  — + стемминг Snowball (PyStemmer), быстро и грубо;
  * "lemma" — + лемматизация pymorphy3 (нормальная форма самого вероятного разбора).

Лемматизация кэшируется по уникальным словам: словарь корпуса на порядки меньше
числа токенов, так что pymorphy вызывается ~сотни тысяч раз, а не десятки миллионов.
"""
import re
from functools import lru_cache

import pymorphy3

TOKEN_RE = re.compile(r"[a-zа-я0-9]+")

_morph = pymorphy3.MorphAnalyzer()
_stemmer = None  # PyStemmer грузим лениво: нужен только в эксперименте 01


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("ё", "е"))


@lru_cache(maxsize=2_000_000)
def lemma(word: str) -> str:
    # цифры и латиницу pymorphy не улучшит — оставляем как есть
    if not ("а" <= word[0] <= "я"):
        return word
    return _morph.parse(word)[0].normal_form.replace("ё", "е")


@lru_cache(maxsize=2_000_000)
def stem(word: str) -> str:
    global _stemmer
    if _stemmer is None:
        import Stemmer

        _stemmer = Stemmer.Stemmer("russian")
    return _stemmer.stemWord(word)


def normalize(text: str, mode: str = "lemma") -> list[str]:
    toks = tokenize(text)
    if mode == "lemma":
        return [lemma(t) for t in toks]
    if mode == "stem":
        return [stem(t) for t in toks]
    return toks
