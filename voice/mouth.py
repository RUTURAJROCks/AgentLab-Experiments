"""Text to speech with Kokoro-82M (Apache 2.0), 24 kHz output.
American English pipeline 'a'; voices: af_heart, af_bella, am_michael, ..."""
import numpy as np


class Mouth:
    SR = 24000

    def __init__(self, voice="af_heart"):
        from kokoro import KPipeline
        self.pipe, self.voice = KPipeline(lang_code="a"), voice

    def say(self, text):
        chunks = [a for _, _, a in self.pipe(text, voice=self.voice)]
        return np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
