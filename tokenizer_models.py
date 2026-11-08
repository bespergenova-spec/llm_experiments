"""Опыт 6: те же фразы (опыт 1) через токенайзеры разных свежих моделей, включая Anthropic.

Источники токенайзеров:
- OpenAI GPT-4 / GPT-4o, 4.1, 5, o-серия: tiktoken (cl100k_base, o200k_base)
- OpenAI gpt-oss, Qwen3, DeepSeek-V3, Mistral Nemo, Gemma 3, Llama 3.1: tokenizer.json с Hugging Face
- Anthropic (старый Claude 1/2): tokenizer.json Xenova/claude-tokenizer — это НЕ токенайзер текущих Claude
- Anthropic (текущие Claude): публичного токенайзера нет, счёт идёт только через API
  (POST /v1/messages/count_tokens). Делается, если задана переменная ANTHROPIC_API_KEY, иначе пропускается.

Запуск: python tokenizer_models.py
Результаты: results/tokenizer_models.json и results/tokenizer_models.md
"""
import json
import os
import urllib.request
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from tokenizer_ru_vs_en import LANGS, PHRASES

BASE = "en"
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

# (подпись, тип, источник)
MODELS = [
    ("OpenAI GPT-4 (cl100k_base)", "tiktoken", "cl100k_base"),
    ("OpenAI GPT-4o / 4.1 / 5 / o-серия (o200k_base)", "tiktoken", "o200k_base"),
    ("OpenAI gpt-oss", "hf", "openai/gpt-oss-20b"),
    ("Anthropic Claude 1/2 (старый)", "hf", "Xenova/claude-tokenizer"),
    ("Qwen3", "hf", "Qwen/Qwen3-8B"),
    ("DeepSeek-V3", "hf", "deepseek-ai/DeepSeek-V3"),
    ("Mistral Nemo (Tekken)", "hf", "mistralai/Mistral-Nemo-Instruct-2407"),
    ("Gemma 3", "hf", "unsloth/gemma-3-4b-it"),
    ("Llama 3.1", "hf", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
    # более новые модели
    ("Llama 4 Scout", "hf", "unsloth/Llama-4-Scout-17B-16E-Instruct"),
    ("Qwen3.5", "hf", "Qwen/Qwen3.5-397B-A17B"),
    ("DeepSeek-V3.2", "hf", "deepseek-ai/DeepSeek-V3.2"),
    ("GLM-5 (Z.ai)", "hf", "zai-org/GLM-5"),
    ("Gemma 4", "hf", "google/gemma-4-31B-it"),
    ("MiniMax-M2", "hf", "MiniMaxAI/MiniMax-M2"),
    ("OLMo 3", "hf", "allenai/Olmo-3-7B-Instruct"),
    ("NVIDIA Nemotron Nano v2", "hf", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
    ("IBM Granite 4.0", "hf", "ibm-granite/granite-4.0-h-small"),
    ("Microsoft Phi-4", "hf", "microsoft/phi-4"),
]


def make_counter(kind, src):
    if kind == "tiktoken":
        enc = tiktoken.get_encoding(src)
        return lambda t: len(enc.encode(t))
    path = hf_hub_download(src, "tokenizer.json")
    tok = Tokenizer.from_file(path)
    # без служебных токенов (BOS и т.п.), чтобы считать только сам текст
    return lambda t: len(tok.encode(t, add_special_tokens=False).ids)


def claude_counter():
    """Счёт токенов через API Anthropic. Возвращает None, если нет ключа."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None

    def raw(text):
        body = json.dumps(
            {"model": CLAUDE_MODEL, "messages": [{"role": "user", "content": text}]}
        ).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages/count_tokens",
            data=body,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["input_tokens"]

    # API считает и служебную обёртку сообщения; вычитаем её по пустому образцу «a» (1 токен текста)
    overhead = raw("a") - 1
    return lambda t: raw(t) - overhead


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    counters = []
    for label, kind, src in MODELS:
        try:
            counters.append((label, src, make_counter(kind, src), None))
        except Exception as e:  # модель недоступна — фиксируем и идём дальше
            counters.append((label, src, None, f"{type(e).__name__}: {e}"))
    cc = claude_counter()
    if cc:
        counters.append((f"Anthropic {CLAUDE_MODEL} (API count_tokens)", CLAUDE_MODEL, cc, None))
    else:
        counters.append(
            (f"Anthropic {CLAUDE_MODEL} (API count_tokens)", CLAUDE_MODEL, None, "пропущено: нет ANTHROPIC_API_KEY")
        )

    results = []
    for label, src, counter, err in counters:
        res = {"model": label, "source": src, "error": err, "phrases": {}, "totals": {}}
        if counter:
            for lang in LANGS:
                res["totals"][lang] = 0
            for p in PHRASES:
                res["phrases"][p["name"]] = {}
                for lang in LANGS:
                    n = counter(p[lang])
                    res["phrases"][p["name"]][lang] = n
                    res["totals"][lang] += n
        results.append(res)

    (out / "tokenizer_models.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = ["# Опыт 6: токенайзеры разных моделей на тех же фразах\n"]
    md.append(
        "Сумма токенов по трём фразам из опыта 1 (pangram, greeting, про токен). "
        "В скобках — во сколько раз больше, чем в английском у той же модели. "
        "Служебные токены (BOS и т.п.) не считаются.\n"
    )
    md.append("| Модель | " + " | ".join(LANGS.values()) + " |")
    md.append("|---|" + "---|" * len(LANGS))
    for r in results:
        if r["error"]:
            md.append(f"| {r['model']} | {r['error']} |" + " |" * (len(LANGS) - 1))
            continue
        en = r["totals"][BASE]
        cells = [str(en) if l == BASE else f"{r['totals'][l]} ({r['totals'][l] / en:.2f}x)" for l in LANGS]
        md.append(f"| {r['model']} | " + " | ".join(cells) + " |")

    md.append("\n## По фразам (EN / RU / RU-лат / KK)\n")
    md.append("| Модель | pangram | greeting | про токен |")
    md.append("|---|---|---|---|")
    for r in results:
        if r["error"]:
            continue
        cells = []
        for p in PHRASES:
            v = r["phrases"][p["name"]]
            cells.append(" / ".join(str(v[l]) for l in LANGS))
        md.append(f"| {r['model']} | " + " | ".join(cells) + " |")

    md.append("\n## Источники\n")
    for r in results:
        md.append(f"- {r['model']}: `{r['source']}`")
    (out / "tokenizer_models.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
