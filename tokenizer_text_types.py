"""Опыт 3: как токенайзер режет разные типы текста (слова, числа, код, URL, эмодзи, пробелы, имена).

Показывает, почему модели трудно считать буквы и делать арифметику «в уме».

Запуск: python tokenizer_text_types.py
Результаты: results/tokenizer_text_types.json и results/tokenizer_text_types.md
"""
import json
import re
from pathlib import Path

import tiktoken

SAMPLES = [
    ("слово и регистр", "strawberry"),
    ("слово и регистр", " strawberry"),
    ("слово и регистр", "Strawberry"),
    ("слово и регистр", "STRAWBERRY"),
    ("слово и регистр", "s t r a w b e r r y"),
    ("имя", "Balnur"),
    ("имя", "Балнур"),
    ("числа", "1234567890"),
    ("числа", "3.14159265"),
    ("числа", "1 000 000"),
    ("числа", "2026-11-08"),
    ("арифметика", "123456 + 654321 = 777777"),
    ("код", "def hello():\n    return 42"),
    ("JSON", '{"name": "Балнур", "age": 30}'),
    ("URL", "https://github.com/bespergenova-spec/llm_experiments"),
    ("эмодзи", "☕"),
    ("эмодзи", "😀"),
    ("эмодзи", "👨‍👩‍👧"),
    ("пробелы", "    "),
    ("пробелы", "\n\n\n"),
]
ENCODINGS = ["cl100k_base", "o200k_base"]


def split(enc, text):
    ids = enc.encode(text)
    pieces = []
    for i in ids:
        b = enc.decode_single_token_bytes(i)
        try:
            pieces.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            pieces.append("<" + " ".join(f"{x:02X}" for x in b) + ">")
    return ids, pieces


def show(s):
    """Делает пробелы и переводы строк видимыми для таблицы (hex-байты не трогаем)."""
    if re.fullmatch(r"<[0-9A-F]{2}( [0-9A-F]{2})*>", s):
        return s
    return s.replace("\n", "\\n").replace(" ", "·")


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    encs = {n: tiktoken.get_encoding(n) for n in ENCODINGS}

    rows = []
    for category, text in SAMPLES:
        row = {
            "category": category,
            "text": text,
            "chars": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
        }
        for name, enc in encs.items():
            ids, pieces = split(enc, text)
            row[name] = {"tokens": len(ids), "pieces": pieces}
        rows.append(row)

    (out / "tokenizer_text_types.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = ["# Опыт 3: как режутся разные типы текста\n"]
    md.append("Пробел показан как `·`, перевод строки как `\\n`. `<D0 9A>` — токен, содержащий часть байтов символа.\n")
    md.append("| Тип | Текст | Симв. | Байт | cl100k | o200k |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['category']} | `{show(r['text'])}` | {r['chars']} | {r['utf8_bytes']} | "
            f"{r['cl100k_base']['tokens']} | {r['o200k_base']['tokens']} |"
        )
    md.append("\n## Разбивка на токены\n")
    for r in rows:
        md.append(f"### {r['category']}: `{show(r['text'])}`\n")
        for name in ENCODINGS:
            pieces = " ".join(f"[{show(s)}]" for s in r[name]["pieces"])
            md.append(f"- **{name}** ({r[name]['tokens']}): `{pieces}`")
        md.append("")
    (out / "tokenizer_text_types.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
