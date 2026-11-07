"""Опыт 1: одна и та же фраза на разных языках/письменностях через токенайзеры tiktoken.

Варианты: английский (база), русский, русский латиницей (транслит), казахский (кириллица).

Запуск: python tokenizer_ru_vs_en.py
Результаты: results/tokenizer_ru_vs_en.json и results/tokenizer_ru_vs_en.md
"""
import json
from pathlib import Path

import tiktoken

LANGS = {
    "en": "английский",
    "ru": "русский",
    "ru_lat": "русский латиницей",
    "kk": "казахский",
}
BASE = "en"

PHRASES = [
    {
        "name": "pangram",
        "en": "The quick brown fox jumps over the lazy dog.",
        "ru": "Быстрая коричневая лиса перепрыгивает через ленивую собаку.",
        "ru_lat": "Bystraya korichnevaya lisa pereprygivaet cherez lenivuyu sobaku.",
        "kk": "Жылдам қоңыр түлкі жалқау итті аттап өтеді.",
    },
    {
        "name": "greeting",
        "en": "Hello, world! How are you today?",
        "ru": "Привет, мир! Как у тебя дела сегодня?",
        "ru_lat": "Privet, mir! Kak u tebya dela segodnya?",
        "kk": "Сәлем, әлем! Бүгін қалайсың?",
    },
    {
        "name": "tokenization",
        "en": "A token is a piece of text that a language model works with.",
        "ru": "Токен — это кусочек текста, с которым работает языковая модель.",
        "ru_lat": "Token — eto kusochek teksta, s kotorym rabotaet yazykovaya model.",
        "kk": "Токен — тілдік модель жұмыс істейтін мәтін бөлігі.",
    },
]
ENCODINGS = ["cl100k_base", "o200k_base"]


def split(enc, text):
    ids = enc.encode(text)
    # Токен может содержать лишь часть байтов UTF-8 буквы (часто у казахского в cl100k):
    # такие токены показываем как hex-байты <D2 9B>, а не как мусорный символ.
    pieces = []
    for i in ids:
        b = enc.decode_single_token_bytes(i)
        try:
            pieces.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            pieces.append("<" + " ".join(f"{x:02X}" for x in b) + ">")
    return ids, pieces


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        for p in PHRASES:
            row = {"encoding": enc_name, "phrase": p["name"]}
            for lang in LANGS:
                text = p[lang]
                ids, pieces = split(enc, text)
                row[lang] = {
                    "text": text,
                    "chars": len(text),
                    "utf8_bytes": len(text.encode("utf-8")),
                    "tokens": len(ids),
                    "chars_per_token": round(len(text) / len(ids), 2),
                    "ratio_to_en": round(len(ids) / row[BASE]["tokens"], 2) if lang != BASE else 1.0,
                    "pieces": pieces,
                }
            rows.append(row)

    (out / "tokenizer_ru_vs_en.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = ["# Опыт 1: токенизация одной фразы на разных языках\n"]
    md.append("Число токенов (в скобках — во сколько раз больше, чем в английском):\n")
    md.append("| Токенайзер | Фраза | " + " | ".join(LANGS.values()) + " |")
    md.append("|---|---|" + "---|" * len(LANGS))
    for r in rows:
        cells = [
            f"{r[l]['tokens']}" if l == BASE else f"{r[l]['tokens']} ({r[l]['ratio_to_en']}x)"
            for l in LANGS
        ]
        md.append(f"| {r['encoding']} | {r['phrase']} | " + " | ".join(cells) + " |")

    md.append("\nСимволов на токен (чем меньше, тем мельче дробится текст):\n")
    md.append("| Токенайзер | Фраза | " + " | ".join(LANGS.values()) + " |")
    md.append("|---|---|" + "---|" * len(LANGS))
    for r in rows:
        md.append(
            f"| {r['encoding']} | {r['phrase']} | "
            + " | ".join(str(r[l]["chars_per_token"]) for l in LANGS)
            + " |"
        )

    md.append("\n## Разбивка на токены\n")
    for r in rows:
        md.append(f"### {r['encoding']} / {r['phrase']}\n")
        for lang, label in LANGS.items():
            pieces = " ".join(f"[{s}]" for s in r[lang]["pieces"])
            md.append(f"- **{label}** ({r[lang]['tokens']}): `{pieces}`")
        md.append("")
    (out / "tokenizer_ru_vs_en.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
