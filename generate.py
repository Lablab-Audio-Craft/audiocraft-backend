from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os
import loguru
import scipy
import torch

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda", 0)


logger = loguru.logger


def main(prompt):
    logger.info("generating audio from prompt")
    outdir = "static/"
    length = os.listdir(outdir)
    outpath = f"{outdir}combined_audio_{len(length)+1}.wav"
    processor = AutoProcessor.from_pretrained("models/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("models/musicgen-medium")
    model.to(device)

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    audio_values = model.generate(**inputs, max_new_tokens=512)
    audio_values = audio_values.cpu()
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(outpath, rate=sampling_rate, data=audio_values[0, 0].numpy())
    return outpath


if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    main(prompt)
