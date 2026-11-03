# agentlab voice (agent 09)

Talk to Gemma 4 E4B, fully local, on a free Colab T4. English only for now.

Pipeline: your mic -> Silero VAD + faster-whisper large-v3-turbo -> Gemma 4 E4B-it (4-bit) -> Kokoro-82M voice.

## Three listening modes (the experiment)
| mode | what Gemma gets |
| --- | --- |
| native | the raw audio only (Gemma 4's own audio encoder) |
| cascade | the Whisper transcript only (the classic pipeline) |
| hybrid | audio + transcript, told to trust the audio where they disagree |

## What we do to score well (all from public resources, all disclosed)
- Strong open parts: Whisper large-v3-turbo + Silero VAD, Kokoro-82M, Gemma 4 E4B.
- Hybrid input: lets the model fix ASR errors from the sound, and mishearing from the text.
- Thinking on for reasoning sets (OpenBookQA, MMSU, BBH), off for chat to keep latency low.
- Spoken-style system prompt: answer first, short, follow format instructions exactly, end multiple-choice with "The answer is (X)."
- Greedy decoding in benchmarks so runs repeat.

## Benchmark
VoiceBench (TACL 2026), scored with its own evaluators so numbers match its leaderboard:
https://matthewcym.github.io/VoiceBench (dataset hlt-lab/voicebench).
Rule-scored subsets run free: openbookqa, mmsu, ifeval, advbench, bbh.
Judge-scored subsets (alpacaeval_full, commoneval, wildvoice, sd-qa) need an OpenAI key for api_judge.py.

Reference points on the leaderboard (overall): Parakeet-TDT-0.6b-V2 + Qwen3-8B cascade 79.23; Whisper-v3-turbo + LLaMA-3.2-3B 71.02; Whisper-v3-large + GPT-4o 87.80; top open: BR-Voice-Reasoner 90.46.

## Run
Open voice_colab.ipynb in Colab, Runtime -> T4 GPU, Run all.
Locally: pip install -r voice/requirements.txt, then python -m voice.app or python -m voice.bench --mode hybrid --limit 100

Not yet run on a GPU by us: first Colab run is the real test. If Gemma 4 audio gives garbage in float16 on T4, try VOICE_MODEL=google/gemma-4-E2B-it or four_bit=False on a bigger GPU.
