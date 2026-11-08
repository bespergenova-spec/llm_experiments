"""Опыт 5: top-k и top-p (nucleus sampling). Симуляция на тех же иллюстративных логитах, что в опыте 4.

Порядок, как в большинстве реализаций: сначала температура (softmax), затем отсечение
хвоста (top-k или top-p), затем перенормировка оставшихся вероятностей и выбор.

Запуск: python topk_topp.py
Результаты: results/topk_topp.json и results/topk_topp.md
"""
import json
import random
from collections import Counter
from pathlib import Path

from temperature_softmax import LOGITS, bar, entropy_bits, softmax

TEMPERATURES = [1.0, 1.5]
TOP_K = [1, 3, 5, len(LOGITS)]
TOP_P = [0.3, 0.5, 0.8, 0.9, 0.95, 1.0]
N_SAMPLES = 1000
SEED = 42


def apply_top_k(p, k):
    keep = sorted(p, key=p.get, reverse=True)[:k]
    return renorm(p, keep)


def apply_top_p(p, top_p):
    """Оставляем минимальный набор самых вероятных вариантов с суммой >= top_p."""
    keep, total = [], 0.0
    for n in sorted(p, key=p.get, reverse=True):
        keep.append(n)
        total += p[n]
        if total >= top_p - 1e-12:
            break
    return renorm(p, keep)


def renorm(p, keep):
    z = sum(p[n] for n in keep)
    return {n: (p[n] / z if n in keep else 0.0) for n in p}


def stats(p, rng):
    names = list(p)
    draws = rng.choices(names, weights=[p[n] for n in names], k=N_SAMPLES)
    counts = Counter(draws)
    return {
        "kept": sum(1 for x in p.values() if x > 0),
        "entropy_bits": round(entropy_bits(p), 3),
        "top_share": round(max(p.values()), 3),
        "unique_in_1000": len(counts),
        "p_three_same": round(sum(x**3 for x in p.values()), 4),
        "expected_unique_of_3": round(sum(1 - (1 - x) ** 3 for x in p.values()), 2),
        "tail_draws": sum(c for n, c in counts.items() if n in TAIL),
    }


# «Хвост» — три самых маловероятных варианта (по логитам)
TAIL = set(sorted(LOGITS, key=LOGITS.get)[:3])


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    results = []
    for t in TEMPERATURES:
        base = softmax(LOGITS, t)
        configs = [("без отсечения", base)]
        configs += [(f"top-k={k}", apply_top_k(base, k)) for k in TOP_K if k < len(LOGITS)]
        configs += [(f"top-p={tp}", apply_top_p(base, tp)) for tp in TOP_P if tp < 1.0]
        for label, p in configs:
            results.append(
                {
                    "temperature": t,
                    "method": label,
                    "probabilities": {n: round(v, 4) for n, v in p.items()},
                    **stats(p, rng),
                }
            )

    (out / "topk_topp.json").write_text(
        json.dumps(
            {"logits": LOGITS, "tail": sorted(TAIL), "n_samples": N_SAMPLES, "seed": SEED, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = ["# Опыт 5: top-k и top-p\n"]
    md.append(
        "Логиты те же иллюстративные, что в опыте 4 (не от реальной модели). "
        f"Выборка: {N_SAMPLES} на каждую конфигурацию, seed={SEED}. "
        f"«Хвост» = три самых редких варианта: {', '.join(sorted(TAIL))}.\n"
    )
    md.append("- **top-k=k** — оставить k самых вероятных вариантов.")
    md.append("- **top-p=p** — оставить минимальный набор самых вероятных, суммарная вероятность которого ≥ p. Размер набора зависит от формы распределения.")
    md.append("- После отсечения вероятности перенормируются, затем идёт выбор.\n")

    for t in TEMPERATURES:
        md.append(f"## Температура {t}\n")
        md.append("| Метод | Осталось вариантов | Энтропия, бит | Доля лидера | Разных из 1000 | Попаданий в хвост из 1000 | Шанс, что 3 чата совпадут | В среднем разных из 3 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in results:
            if r["temperature"] != t:
                continue
            md.append(
                f"| {r['method']} | {r['kept']} | {r['entropy_bits']} | {r['top_share'] * 100:.1f}% | "
                f"{r['unique_in_1000']} | {r['tail_draws']} | {r['p_three_same'] * 100:.1f}% | {r['expected_unique_of_3']} |"
            )
        md.append("")

    md.append("## Вероятности после отсечения при T=1.0\n")
    t1 = [r for r in results if r["temperature"] == 1.0]
    md.append("| Название | " + " | ".join(r["method"] for r in t1) + " |")
    md.append("|---|" + "---|" * len(t1))
    for n in LOGITS:
        md.append(f"| {n} | " + " | ".join(f"{r['probabilities'][n] * 100:.1f}%" for r in t1) + " |")

    md.append("\n## Гистограмма: top-p=0.9 против без отсечения (T=1.5)\n")
    md.append("```")
    t15 = {r["method"]: r for r in results if r["temperature"] == 1.5}
    for n in LOGITS:
        a = t15["без отсечения"]["probabilities"][n]
        b = t15["top-p=0.9"]["probabilities"][n]
        md.append(f"{n:<16} без: {a * 100:5.1f}% {bar(a, 20):<20} | top-p 0.9: {b * 100:5.1f}% {bar(b, 20)}")
    md.append("```")

    (out / "topk_topp.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
