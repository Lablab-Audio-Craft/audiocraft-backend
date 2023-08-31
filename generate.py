from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import torch
import os


def get_outpath():
    count = len(os.listdir("./out"))
    return f"./out/audio_{count}.wav"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("models/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("models/musicgen-medium")

inputs = processor(
    text=["uplifting trance track with driving bass and uplifting synth plucks"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, max_new_tokens=256)

out_path = get_outpath()

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write(out_path, rate=sampling_rate, data=audio_values[0, 0].numpy())
