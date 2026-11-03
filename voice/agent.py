"""Voice agent 09: talk to Gemma 4 locally. English, free Colab T4.

Three ways to listen, so we can measure which one scores best:
  native  : Gemma 4 E4B hears the raw audio itself (its ~300M audio encoder)
  cascade : faster-whisper transcribes, Gemma reads the text
  hybrid  : Gemma gets the audio AND the Whisper transcript, so it can
            fix ASR mistakes from the sound and fix mishearing from the text
Then Kokoro-82M speaks the reply, sentence by sentence.

Model card notes we follow (huggingface.co/google/gemma-4-E4B-it):
  - audio goes AFTER the text in the user turn, max 30 s per clip
  - thinking is switched on with enable_thinking=True
  - recommended sampling: temperature 1.0, top_p 0.95, top_k 64
    (we decode greedily for benchmarks so runs are repeatable)
"""
import os, re, tempfile, threading, time
import numpy as np
import soundfile as sf
import torch

MODEL_ID = os.getenv("VOICE_MODEL", "google/gemma-4-E4B-it")
MAX_CLIP_S = 30

SYSTEM = (
    "You are a friendly voice assistant. The user is speaking to you and your reply "
    "will be read aloud. Answer the question first, in plain spoken English: short "
    "sentences, no markdown, no lists, no emojis. Keep chat replies under 60 words "
    "unless the user asks for more. Follow every instruction about format, length or "
    "wording exactly. If you are given a multiple-choice question, reason briefly, "
    "then end with 'The answer is (X).' Refuse clearly if a request is harmful."
)

_THINK_HINTS = ("option", "which of", "choose", "true or false", "how many",
                "calculate", "if ", "puzzle", "order", "navigate", "lie", "truth")


def _clean(text):
    """Drop Gemma 4 thought blocks and control tokens from a decoded reply."""
    text = re.sub(r"<\|channel\>thought.*?<channel\|>", "", text, flags=re.S)
    text = re.sub(r"<\|?[a-z_]+\|?>", "", text)
    return text.replace("<eos>", "").strip()


class VoiceAgent:
    def __init__(self, mode="hybrid", model_id=MODEL_ID, four_bit=True,
                 asr_model="large-v3-turbo", tts_voice="af_heart"):
        from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
        self.mode, self.history = mode, []
        kw = dict(device_map="auto")
        if four_bit:   # fits a 16 GB T4 next to Whisper and Kokoro
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["dtype"] = "auto"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **kw).eval()
        self.ears = None
        if mode in ("cascade", "hybrid"):
            from .ears import Ears
            self.ears = Ears(asr_model)
        self.tts_voice, self.mouth = tts_voice, None

    # ---------- listening ----------
    def _turn(self, audio, sr, text=None):
        """Build the user turn. audio: float32 mono numpy, sr: its sample rate."""
        transcript = None
        if audio is not None and self.ears:
            transcript = self.ears.transcribe(audio, sr)
        content = []
        if self.mode == "cascade" or audio is None:
            content.append({"type": "text", "text": text or transcript or ""})
            return content, transcript
        note = ("Listen to the user's spoken message and reply to it."
                if self.mode == "native" else
                f"Listen to the user's spoken message and reply to it. A speech "
                f"recogniser heard: \"{transcript}\". It may contain errors; trust "
                f"the audio where they disagree.")
        audio = audio[: int(sr * MAX_CLIP_S)]
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(f.name, audio, sr)
        content += [{"type": "text", "text": note}, {"type": "audio", "audio": f.name}]
        return content, transcript

    def _think(self, think, transcript):
        if think == "auto":
            return bool(transcript) and any(h in transcript.lower() for h in _THINK_HINTS)
        return bool(think)

    def _inputs(self, content, think, keep_history):
        msgs = [{"role": "system", "content": SYSTEM}]
        if keep_history:
            msgs += self.history
        msgs.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=think).to(self.model.device)

    # ---------- replying ----------
    @torch.no_grad()
    def reply(self, audio=None, sr=16000, text=None, think="auto",
              max_new_tokens=512, keep_history=True):
        """One full turn, no streaming. Returns (reply_text, transcript)."""
        content, transcript = self._turn(audio, sr, text)
        think = self._think(think, transcript)
        inputs = self._inputs(content, think, keep_history)
        n = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens + (1024 if think else 0),
                                  do_sample=False)
        answer = _clean(self.processor.decode(out[0][n:], skip_special_tokens=False))
        if keep_history:
            self._remember(transcript or text or "(spoken message)", answer)
        return answer, transcript

    @torch.no_grad()
    def stream(self, audio=None, sr=16000, text=None, think=False, max_new_tokens=300):
        """Yield (sentence, audio_chunk_24k) as soon as each sentence is ready."""
        from transformers import TextIteratorStreamer
        if self.mouth is None:
            from .mouth import Mouth
            self.mouth = Mouth(self.tts_voice)
        content, transcript = self._turn(audio, sr, text)
        inputs = self._inputs(content, self._think(think, transcript), True)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True,
                                        skip_special_tokens=False)
        threading.Thread(target=self.model.generate, kwargs=dict(
            **inputs, streamer=streamer, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.95, top_k=64)).start()
        buf, full, in_thought = "", "", False
        for piece in streamer:
            full += piece
            if "<|channel>thought" in piece: in_thought = True
            if in_thought:
                if "<channel|>" in piece: in_thought = False
                continue
            buf += piece
            parts = re.split(r"(?<=[.!?])\s+", _clean(buf))
            for s in parts[:-1]:
                if s.strip():
                    yield s, self.mouth.say(s)
            buf = parts[-1]
        if _clean(buf):
            yield _clean(buf), self.mouth.say(_clean(buf))
        self._remember(transcript or text or "(spoken message)", _clean(full))

    def _remember(self, user, answer):
        # history is text only: keeps context small and follows the card's
        # rule of never putting old thoughts back into the conversation
        self.history += [{"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]
        self.history = self.history[-12:]

    def reset(self):
        self.history = []
