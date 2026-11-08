# Опыт 6: токенайзеры разных моделей на тех же фразах

Сумма токенов по трём фразам из опыта 1 (pangram, greeting, про токен). В скобках — во сколько раз больше, чем в английском у той же модели. Служебные токены (BOS и т.п.) не считаются.

| Модель | английский | русский | русский латиницей | казахский |
|---|---|---|---|---|
| OpenAI GPT-4 (cl100k_base) | 33 | 76 (2.30x) | 63 (1.91x) | 104 (3.15x) |
| OpenAI GPT-4o / 4.1 / 5 / o-серия (o200k_base) | 33 | 48 (1.45x) | 57 (1.73x) | 46 (1.39x) |
| OpenAI gpt-oss | 33 | 48 (1.45x) | 57 (1.73x) | 46 (1.39x) |
| Anthropic Claude 1/2 (старый) | 33 | 84 (2.55x) | 72 (2.18x) | 107 (3.24x) |
| Qwen3 | 33 | 56 (1.70x) | 63 (1.91x) | 88 (2.67x) |
| DeepSeek-V3 | 33 | 54 (1.64x) | 66 (2.00x) | 68 (2.06x) |
| Mistral Nemo (Tekken) | 33 | 51 (1.55x) | 61 (1.85x) | 57 (1.73x) |
| Gemma 3 | 33 | 43 (1.30x) | 59 (1.79x) | 57 (1.73x) |
| Llama 3.1 | 33 | 54 (1.64x) | 61 (1.85x) | 85 (2.58x) |
| Llama 4 Scout | 33 | 46 (1.39x) | 60 (1.82x) | 53 (1.61x) |
| Qwen3.5 | 33 | 50 (1.52x) | 60 (1.82x) | 62 (1.88x) |
| DeepSeek-V3.2 | 33 | 54 (1.64x) | 66 (2.00x) | 68 (2.06x) |
| GLM-5 (Z.ai) | 33 | 49 (1.48x) | 63 (1.91x) | 74 (2.24x) |
| Gemma 4 | 33 | 43 (1.30x) | 59 (1.79x) | 57 (1.73x) |
| MiniMax-M2 | 33 | 58 (1.76x) | 59 (1.79x) | 76 (2.30x) |
| OLMo 3 | 33 | 76 (2.30x) | 63 (1.91x) | 104 (3.15x) |
| NVIDIA Nemotron Nano v2 | 33 | 51 (1.55x) | 61 (1.85x) | 57 (1.73x) |
| IBM Granite 4.0 | 33 | 76 (2.30x) | 63 (1.91x) | 104 (3.15x) |
| Microsoft Phi-4 | 33 | 76 (2.30x) | 63 (1.91x) | 104 (3.15x) |
| Anthropic claude-sonnet-5-5 (API count_tokens) | пропущено: нет ANTHROPIC_API_KEY | | | |

## По фразам (EN / RU / RU-лат / KK)

| Модель | pangram | greeting | про токен |
|---|---|---|---|
| OpenAI GPT-4 (cl100k_base) | 10 / 30 / 24 / 41 | 9 / 20 / 15 / 25 | 14 / 26 / 24 / 38 |
| OpenAI GPT-4o / 4.1 / 5 / o-серия (o200k_base) | 10 / 20 / 21 / 18 | 9 / 11 / 14 / 13 | 14 / 17 / 22 / 15 |
| OpenAI gpt-oss | 10 / 20 / 21 / 18 | 9 / 11 / 14 / 13 | 14 / 17 / 22 / 15 |
| Anthropic Claude 1/2 (старый) | 10 / 33 / 27 / 41 | 9 / 22 / 18 / 26 | 14 / 29 / 27 / 40 |
| Qwen3 | 10 / 24 / 24 / 35 | 9 / 12 / 15 / 20 | 14 / 20 / 24 / 33 |
| DeepSeek-V3 | 10 / 23 / 26 / 24 | 9 / 12 / 15 / 18 | 14 / 19 / 25 / 26 |
| Mistral Nemo (Tekken) | 10 / 22 / 23 / 22 | 9 / 11 / 15 / 15 | 14 / 18 / 23 / 20 |
| Gemma 3 | 10 / 18 / 21 / 20 | 9 / 10 / 15 / 15 | 14 / 15 / 23 / 22 |
| Llama 3.1 | 10 / 24 / 23 / 31 | 9 / 12 / 15 / 23 | 14 / 18 / 23 / 31 |
| Llama 4 Scout | 10 / 18 / 22 / 21 | 9 / 11 / 15 / 15 | 14 / 17 / 23 / 17 |
| Qwen3.5 | 10 / 21 / 22 / 21 | 9 / 12 / 15 / 17 | 14 / 17 / 23 / 24 |
| DeepSeek-V3.2 | 10 / 23 / 26 / 24 | 9 / 12 / 15 / 18 | 14 / 19 / 25 / 26 |
| GLM-5 (Z.ai) | 10 / 20 / 24 / 28 | 9 / 11 / 15 / 19 | 14 / 18 / 24 / 27 |
| Gemma 4 | 10 / 18 / 21 / 20 | 9 / 10 / 15 / 15 | 14 / 15 / 23 / 22 |
| MiniMax-M2 | 10 / 24 / 21 / 29 | 9 / 12 / 15 / 19 | 14 / 22 / 23 / 28 |
| OLMo 3 | 10 / 30 / 24 / 41 | 9 / 20 / 15 / 25 | 14 / 26 / 24 / 38 |
| NVIDIA Nemotron Nano v2 | 10 / 22 / 23 / 22 | 9 / 11 / 15 / 15 | 14 / 18 / 23 / 20 |
| IBM Granite 4.0 | 10 / 30 / 24 / 41 | 9 / 20 / 15 / 25 | 14 / 26 / 24 / 38 |
| Microsoft Phi-4 | 10 / 30 / 24 / 41 | 9 / 20 / 15 / 25 | 14 / 26 / 24 / 38 |

## Источники

- OpenAI GPT-4 (cl100k_base): `cl100k_base`
- OpenAI GPT-4o / 4.1 / 5 / o-серия (o200k_base): `o200k_base`
- OpenAI gpt-oss: `openai/gpt-oss-20b`
- Anthropic Claude 1/2 (старый): `Xenova/claude-tokenizer`
- Qwen3: `Qwen/Qwen3-8B`
- DeepSeek-V3: `deepseek-ai/DeepSeek-V3`
- Mistral Nemo (Tekken): `mistralai/Mistral-Nemo-Instruct-2407`
- Gemma 3: `unsloth/gemma-3-4b-it`
- Llama 3.1: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Llama 4 Scout: `unsloth/Llama-4-Scout-17B-16E-Instruct`
- Qwen3.5: `Qwen/Qwen3.5-397B-A17B`
- DeepSeek-V3.2: `deepseek-ai/DeepSeek-V3.2`
- GLM-5 (Z.ai): `zai-org/GLM-5`
- Gemma 4: `google/gemma-4-31B-it`
- MiniMax-M2: `MiniMaxAI/MiniMax-M2`
- OLMo 3: `allenai/Olmo-3-7B-Instruct`
- NVIDIA Nemotron Nano v2: `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- IBM Granite 4.0: `ibm-granite/granite-4.0-h-small`
- Microsoft Phi-4: `microsoft/phi-4`
- Anthropic claude-sonnet-5-5 (API count_tokens): `claude-sonnet-5-5`