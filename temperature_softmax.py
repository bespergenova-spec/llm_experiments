"""Опыт 4: температура. Симуляция на softmax.

Берём набор «логитов» для выбора названия кофейни и смотрим, как температура
меняет вероятности и разброс выборок.

ВАЖНО: логиты здесь иллюстративные, подобраны по частотам из опыта 2
(results/randomness_coffee_names.md), а не взяты у реальной модели.
Сама математика (softmax с температурой) настоящая.

Запуск: python temperature_softmax.py
Результаты: results/temperature_softmax.json и results/temperature_softmax.md
"""
import json
import math
import random
from collections import Counter
from pathlib import Path

LOGITS = {
    "Зерно и Пар": 4.0,
    "Крема": 3.2,
    "Тёмная обжарка": 2.6,
    "Ристретто": 2.6,
    "Бариста на углу": 2.0,
    "Первый глоток": 1.6,
    "Чёрный кот": 1.4,
    "Кофе-пауза": 1.0,
    "Утро в Чашке": 1.0,
    "Arabica Corner": 0.6,
    "Латте и точка": 0.2,
    "Кофейный Дворик": 0.2,
}
TEMPERATURES = [0, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0]
N_SAMPLES = 1000
SEED = 42


def softmax(logits, t):
    """Вероятности при температуре t. При t=0 — жадный выбор (argmax)."""
    names = list(logits)
    if t == 0:
        best = max(names, key=logits.get)
        return {n: 1.0 if n == best else 0.0 for n in names}
    scaled = [logits[n] / t for n in names]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]  # вычитаем max для устойчивости
    z = sum(exps)
    return {n: e / z for n, e in zip(names, exps)}


def entropy_bits(p):
    return abs(-sum(x * math.log2(x) for x in p.values() if x > 0))  # abs убирает -0.0


def bar(x, width=30):
    return "█" * round(x * width)


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    names = list(LOGITS)

    results = []
    for t in TEMPERATURES:
        p = softmax(LOGITS, t)
        draws = rng.choices(names, weights=[p[n] for n in names], k=N_SAMPLES)
        counts = Counter(draws)
        results.append(
            {
                "temperature": t,
                "probabilities": {n: round(p[n], 4) for n in names},
                "entropy_bits": round(entropy_bits(p), 3),
                "top_share_observed": round(counts.most_common(1)[0][1] / N_SAMPLES, 3),
                "unique_in_1000": len(counts),
                # шанс, что три независимых «чата» дали бы один и тот же ответ
                "p_three_same": round(sum(x**3 for x in p.values()), 4),
                # ожидаемое число разных ответов среди трёх
                "expected_unique_of_3": round(sum(1 - (1 - x) ** 3 for x in p.values()), 2),
                "counts": dict(counts.most_common()),
            }
        )

    (out / "temperature_softmax.json").write_text(
        json.dumps(
            {"logits": LOGITS, "n_samples": N_SAMPLES, "seed": SEED, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = ["# Опыт 4: температура (симуляция на softmax)\n"]
    md.append(
        "Логиты иллюстративные (подобраны по опыту 2), не от реальной модели. "
        f"Выборка: {N_SAMPLES} на каждую температуру, seed={SEED}.\n"
    )
    md.append("Формула: `p_i = exp(z_i / T) / Σ exp(z_j / T)`. При `T → 0` остаётся только самый вероятный вариант, при больших `T` распределение выравнивается.\n")

    md.append("## Вероятности по температурам\n")
    header = "| Название | logit | " + " | ".join(f"T={t}" for t in TEMPERATURES) + " |"
    md.append(header)
    md.append("|---|---|" + "---|" * len(TEMPERATURES))
    for n in names:
        row = " | ".join(f"{r['probabilities'][n] * 100:.1f}%" for r in results)
        md.append(f"| {n} | {LOGITS[n]} | {row} |")

    md.append("\n## Сводка\n")
    md.append("| T | энтропия, бит | доля лидера в 1000 | разных из 1000 | шанс, что 3 чата дадут одно и то же | в среднем разных из 3 |")
    md.append("|---|---|---|---|---|---|")
    for r in results:
        md.append(
            f"| {r['temperature']} | {r['entropy_bits']} | {r['top_share_observed'] * 100:.1f}% | "
            f"{r['unique_in_1000']} | {r['p_three_same'] * 100:.1f}% | {r['expected_unique_of_3']} |"
        )

    md.append("\n## Гистограммы (доли из 1000 выборок)\n")
    for r in results:
        md.append(f"### T = {r['temperature']}\n")
        md.append("```")
        for n in names:
            share = r["counts"].get(n, 0) / N_SAMPLES
            md.append(f"{n:<16} {share * 100:5.1f}%  {bar(share)}")
        md.append("```\n")

    (out / "temperature_softmax.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
