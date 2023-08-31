import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import json
import os


def parse_name_title(filename):
    name_title = filename.replace(".wav", "")
    parsed = name_title.split("-")
    if len(parsed) != 2:
        print(f"Could not parse both name and title from filename: {filename}")
    if len(parsed) == 1:
        return parsed[0], ""
    return (parsed[0], parsed[1]) if len(parsed) == 2 else (None, None)


def play_wav_and_save_description(subfolder):
    files = [
        f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))
    ]
    wav_files = [f for f in files if f.lower().endswith(".wav")]

    if not wav_files:
        print("No WAV files found in the given subfolder.")
        return

    for wav_file in wav_files:
        wav_path = os.path.join(subfolder, wav_file)
        audio = AudioSegment.from_wav(wav_path)

        # Extract some audio information
        sample_rate = audio.frame_rate
        duration = len(audio) / 1000.0  # Duration in seconds
        channels = audio.channels
        # Play 1.5 minutes of the WAV file
        play_duration = min(duration, 260)  # 90 seconds = 1.5 minutes
        play_audio = audio[
            90 * 1000 : int(play_duration * 1000)
        ]  # Convert back to milliseconds
        samples = np.array(play_audio.get_array_of_samples())
        # Reshape the array for multi-channel (if required)
        if channels > 1:
            samples = samples.reshape((-1, channels))
        sd.play(data=samples, samplerate=sample_rate)

        # Parse name and title from filename
        name, title = parse_name_title(wav_file)

        # Prompt for description and other metadata
        description = input(f"{name} - {title}:\nEnter a description for this audio: ")
        keywords = input(
            f"{name} - {title}:\nEnter keywords for this audio (comma-separated): "
        )
        genre = input(f"{name} - {title}:\nEnter the genre for this audio: ")
        instrument = input(f"{name} - {title}:\nEnter the instrument for this audio: ")
        moods = input(
            f"{name} - {title}:\nEnter the moods for this audio (comma-separated): "
        ).split(",")

        # Stop the audio playback
        sd.stop()

        # Prepare JSON data
        metadata = {
            "key": "",
            "artist": "",
            "sample_rate": sample_rate,
            "file_extension": "wav",
            "description": description,
            "keywords": keywords,
            "duration": duration,
            "bpm": "",
            "genre": genre,
            "title": title,
            "name": name,
            "instrument": instrument,
            "moods": moods,
        }

        # Save to JSON file
        json_path = os.path.join(subfolder, wav_file.replace(".wav", ".json"))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Metadata saved for {wav_file}")


# Replace 'your_subfolder' with the actual path to your subfolder
play_wav_and_save_description("dataset/dataset/")
