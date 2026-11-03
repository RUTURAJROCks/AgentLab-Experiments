"""Talk to it: a Gradio page with your mic. On Colab, launch(share=True)
prints a public link; open it on your laptop or phone and allow the mic.
Record, stop, and the reply plays back as it is generated."""
import numpy as np
import gradio as gr
from .agent import VoiceAgent


def build(agent: VoiceAgent):
    def talk(mic, chat):
        if mic is None:
            return
        sr, audio = mic
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.abs(audio).max() > 1.0:
            audio /= 32768.0
        chat = chat + [{"role": "user", "content": "🎙️"}, {"role": "assistant", "content": ""}]
        for sentence, wav in agent.stream(audio=audio, sr=sr):
            chat[-1]["content"] += sentence + " "
            yield chat, (24000, wav)
        if agent.history:
            chat[-2]["content"] = agent.history[-2]["content"]
        yield chat, None

    def clear():
        agent.reset()
        return [], None

    with gr.Blocks(title="agentlab voice") as demo:
        gr.Markdown(f"### agentlab voice · Gemma 4 · mode: {agent.mode}")
        chat = gr.Chatbot(type="messages", height=380)
        mic = gr.Audio(sources=["microphone"], type="numpy", label="Speak, then stop")
        out = gr.Audio(streaming=True, autoplay=True, label="Reply")
        mic.stop_recording(talk, [mic, chat], [chat, out])
        gr.Button("New conversation").click(clear, None, [chat, out])
    return demo


if __name__ == "__main__":
    build(VoiceAgent()).queue().launch(share=True)
