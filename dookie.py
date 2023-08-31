import torch
import torchaudio
from torch import tensor

import loguru
import random
import os
import io
from io import BytesIO
import json
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment
from typing import Union, Optional

logger = loguru.logger

is_cuda_available = torch.cuda.is_available()
SONG_PATH = "in/Off_Kilter_[Master]-Bako-48k-32Bit-1db-1.mp3"
MIN_DUR = 10
MAX_DUR = 20
MODEL_PATH = "facebook/musicgen-small"
BPM = 128
ITERATIONS = 7


def run(
    audio: BytesIO,
    bpm: int,
    iterations: Optional[int] = 4,
    min_dur: Optional[int] = 15,
    max_dur: Optional[int] = 25,
    dur: Optional[int] = None,
):
    if min_dur:
        MIN_DUR = min_dur
    if max_dur:
        MAX_DUR = max_dur
    MODEL_PATH = "facebook/musicgen-small"
    if bpm:
        BPM = bpm
    if iterations:
        ITERATIONS = iterations

    def get_song_path(bytes_array: BytesIO):
        try:
            audio_bytes = bytes_array
            for each in os.listdir("in/"):
                with open(f"in/{each}", "wb") as f:
                    f.write(audio_bytes.read())
                return f"in/{each}"

        except FileNotFoundError as error:
            raise Exception("Error in get_out_path function") from error

    if audio:
        SONG_PATH = get_song_path(audio)

    def get_temp_path(i):
        try:
            TEMP_PATH = f"static/continue_{i}"
            return TEMP_PATH
        except FileNotFoundError as error:
            raise Exception("Error in get_out_path function") from error

    def get_out_path():
        try:
            count = len(os.listdir("static/"))
            OUT_PATH = f"static/combined_audio_{count}.wav"
            return OUT_PATH
        except FileNotFoundError as error:
            raise Exception("Error in get_out_path function") from error

    # Utility Functions
    def peak_normalize(y, target_peak=0.9):
        return target_peak * (y / np.max(np.abs(y)))

    def rms_normalize(y, target_rms=0.05):
        return y * (target_rms / np.sqrt(np.mean(y**2)))

    try:

        def preprocess_audio(waveform):
            logger.info("Preprocessing audio")
            waveform_tensor = tensor(data=waveform, device="cpu")
            waveform_tensor.cpu()
            waveform_np = waveform_tensor.squeeze().numpy()
            processed_waveform_np = rms_normalize(peak_normalize(waveform_np))
            processed_waveform_np = torch.from_numpy(processed_waveform_np).unsqueeze(0)
            return processed_waveform_np

    except Exception as error:
        raise Exception("Error in preprocess_audio function") from error

    try:

        def create_slices(song, sr, slice_duration, num_slices=5):
            logger.info("Creating slices")
            song_length = song.shape[-1] / sr
            slices = []
            for _ in range(num_slices):
                random_start = random.choice(
                    range(
                        0,
                        int((song_length - slice_duration) * sr),
                        int(4 * 60 / 75 * sr),
                    )
                )

                slice_waveform = song[
                    ..., random_start : random_start + int(slice_duration * sr)
                ]

                if len(slice_waveform.squeeze()) < int(slice_duration * sr):
                    additional_samples_needed = int(slice_duration * sr) - len(
                        slice_waveform.squeeze()
                    )
                    slice_waveform = torch.cat(
                        [slice_waveform, song[..., :additional_samples_needed]], dim=-1
                    )

                slices.append(slice_waveform)

                return slices

    except Exception as error:
        raise Exception("Error in create_slices function") from error
    if not MIN_DUR:
        MIN_DUR = 10
    if not MAX_DUR:
        MAX_DUR = 20

    def calculate_duration(bpm_input, min_duration=MIN_DUR, max_duration=MAX_DUR):
        logger.info("Calculating duration")
        single_bar_duration = 4 * 60 / bpm_input
        bars = max(min_duration // single_bar_duration, 1)

        while single_bar_duration * bars < min_duration:
            bars += 1

        full_duration = single_bar_duration * bars

        while full_duration > max_duration and bars > 1:
            bars -= 1
            full_duration = single_bar_duration * bars

        return full_duration

    # Main Code
    # Load the song

    SONG_PATH = get_song_path(audio)

    song, sr = torchaudio.load(SONG_PATH)

    # Create slices from the song
    slices = create_slices(song, sr, 35, num_slices=5)
    duration = dur
    # Calculate the optimal duration
    if not dur:
        duration = calculate_duration(BPM)

    # Load the model
    model_continue = MusicGen.get_pretrained(MODEL_PATH)

    model_continue.set_generation_params(duration=duration)
    if not ITERATIONS:
        ITERATIONS = 1
    n_iterations = ITERATIONS
    all_audio_files = []

    try:
        for i in range(n_iterations):
            logger.info(f"Running iteration {i + 1}")
            slice_idx = i % len(slices)

            print(f"Running iteration {i + 1} using slice {slice_idx}...")

            prompt_waveform = slices[slice_idx][..., : int(5 * sr)]  # 5-second duration
            if is_cuda_available:
                prompt_waveform = prompt_waveform.to("cuda")
            prompt_waveform = preprocess_audio(prompt_waveform)

            output = model_continue.generate_continuation(
                prompt_waveform, prompt_sample_rate=sr, progress=True
            )

            # Make sure the output tensor has at most 2 dimensions
            if len(output.size()) > 2:
                output = output.squeeze()
            TEMP_PATH = get_temp_path(i)
            filename = TEMP_PATH
            audio_write(
                filename,
                output.cpu(),
                model_continue.sample_rate,
                strategy="loudness",
                loudness_compressor=True,
            )
            all_audio_files.append(filename)

    except Exception as error:
        raise Exception("Error in main function") from error

        # Combine all audio files
    combined_audio = AudioSegment.empty()
    OUT_PATH = get_out_path()
    for filename in all_audio_files:
        combined_audio += AudioSegment.from_wav(f"{filename}.wav")
        os.remove(f"{filename}.wav")

    return combined_audio.export(OUT_PATH, format="wav")


def main(
    audio,
    bpm,
    iterations,
    min_dur: Optional[int],
    max_dur: Optional[int],
    dur: Optional[int],
) -> Union[str, bool]:
    run(
        audio,
        bpm,
        iterations,
        min_dur,
        max_dur,
        dur,
    )

    count = len(os.listdir("static"))
    return f"static/combined_audio_{count}.wav"


print("Audio generation completed.")

if __name__ == "__main__":
    with open("in/Prog haus.mp3", "rb") as f:
        audio = io.BytesIO(f.read())
    bpm = 120
    iterations = 1
    min_dur = 10
    max_dur = 20
    dur = None
    main(audio, bpm, iterations, min_dur, max_dur, dur)
