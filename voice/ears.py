"""Speech to text with faster-whisper (CTranslate2). Silero VAD is built in
(vad_filter=True), which trims silence before transcription."""
import numpy as np


class Ears:
    def __init__(self, size="large-v3-turbo"):
        from faster_whisper import WhisperModel
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(size, device=dev,
                                  compute_type="float16" if dev == "cuda" else "int8")

    def transcribe(self, audio, sr=16000):
        if sr != 16000:
            import librosa
            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
        segs, _ = self.model.transcribe(np.asarray(audio, dtype=np.float32), language="en",
                                        beam_size=5, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()
