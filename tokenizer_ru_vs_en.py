"""Опыт 1: одна и та же фраза на русском и английском через токенайзеры tiktoken.

Запуск: python tokenizer_ru_vs_en.py
Результаты: results/tokenizer_ru_vs_en.json и results/tokenizer_ru_vs_en.md
"""
import json
from pathlib import Path

import tiktoken

PHRASES = [
    {
        "name": "pangram",
        "en": "The quick brown fox jumps over the lazy dog.",
        "ru": "Быстрая коричневая лиса перепрыгивает через ленивую собаку.",
    },
    {
        "name": "greeting",
        "en": "Hello, world! How are you today?",
        "ru": "Привет, мир! Как у тебя дела сегодня?",
    },
    {
        "name": "tokenization",
        "en": "A token is a piece of text that a language model works with.",
        "ru": "Токен — это кусочек текста, с которым работает языковая модель.",
    },
]
ENCODINGS = ["cl100k_base", "o200k_base"]


def split(enc, text):
    ids = enc.encode(text)
    # decode каждого токена отдельно; у русских букв токен может быть частью байтов UTF-8
    pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
    return ids, pieces


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        for p in PHRASES:
            row = {"encoding": enc_name, "phrase": p["name"]}
            for lang in ("en", "ru"):
                text = p[lang]
                ids, pieces = split(enc, text)
                row[lang] = {
                    "text": text,
                    "chars": len(text),
                    "tokens": len(ids),
                    "chars_per_token": round(len(text) / len(ids), 2),
                    "pieces": pieces,
                }
            row["ratio_ru_to_en"] = round(row["ru"]["tokens"] / row["en"]["tokens"], 2)
            rows.append(row)

    (out / "tokenizer_ru_vs_en.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = ["# Опыт 1: токенизация RU vs EN\n"]
    md.append("| Токенайзер | Фраза | EN токенов | RU токенов | RU/EN | симв/токен EN | симв/токен RU |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['encoding']} | {r['phrase']} | {r['en']['tokens']} | {r['ru']['tokens']} | "
            f"{r['ratio_ru_to_en']}x | {r['en']['chars_per_token']} | {r['ru']['chars_per_token']} |"
        )
    md.append("\n## Разбивка на токены\n")
    for r in rows:
        md.append(f"### {r['encoding']} / {r['phrase']}\n")
        for lang in ("en", "ru"):
            pieces = " ".join(f"[{s}]" for s in r[lang]["pieces"])
            md.append(f"- **{lang.upper()}** ({r[lang]['tokens']}): `{pieces}`")
        md.append("")
    (out / "tokenizer_ru_vs_en.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
