# Sarvam Prism vs Jev: sources (collected Nov 4, 2026)

## Papers and reports
- Sarvam Prism release blog (benchmarks, calibration, attention mask, training objective): https://www.sarvam.ai/blogs/sarvam-prism
- Sarvam Prism model card + inference code (Apache 2.0): https://huggingface.co/sarvamai/prism
- Sarvam Prism API docs (Decisions API, noul/choice/score): https://docs.sarvam.ai/api/token-factory/models/sarvam-prism
- Evaluating and Benchmarking the System One Model Jev (arXiv 2609.37647): https://arxiv.org/abs/2609.37647
- Introducing System One Models & Jev (TypeSafe): https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Artificial Analysis Decision Index v1.0: https://artificialanalysis.ai/articles/artificial-analysis-decision-index
- Jev Decision Index (37 public benchmarks, HF Space): https://huggingface.co/spaces/multimodalart/jev-decision-index
- What is Jev (OpenRouter, how to call the Decisions API): https://openrouter.ai/blog/insights/what-is-jev/
- Is Jev as accurate as frontier models? (Banking77, Opus 84.4% vs Jev 81.0%): https://openrouter.ai/blog/insights/jev-vs-claude-opus-5-classification/

## X post
- @SarvamForDevs launch post: https://x.com/SarvamForDevs/status/2117625820122652693

## News
- AlphaSignal: https://alphasignal.ai/news/sarvam-s-prism-turns-a-105b-open-model-into-a-44ms-decision-engine
- Simon Willison on Jev: https://simonwillison.net/2026/sep/21/jev/

## Key claims to verify (all vendor-reported)
- Prism: 71.8 avg (Jev index 67.0, AA recreation 65.9, Indian languages 82.5), 44 ms median / 420 ms p95 on one B200.
- Jev 1.14: 64.5 avg in Sarvam's table; reported 196 ms, Sarvam measured 431 ms from its cluster.
- Calibration at >=90% confidence, English: Prism expects 3.3% errors, makes 2.2%; Jev 1.14 expects 1.3%, makes 3.5%.
- Indian languages: Prism 5.4% expected vs 4.4% actual; Jev 3.4% vs 16.1%.
- Shuffled options: Prism 0/2,000 changes (fixed order), Jev 12.3% shuffled, 5.6% on identical re-sends.
- Prices: Prism Rs 4 per 1M input tokens, output free. Jev $0.042 per 1M input, output free.
- Local Prism needs one FP8 GPU with ~120 GB (B200/H200), ~106 GB disk: not runnable on free Colab; use the API.
