from pydub import AudioSegment
import os


def normalize_audio_in_folder(subfolder):
    files = [
        f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))
    ]
    wav_files = [f for f in files if f.lower().endswith(".wav")]

    if not wav_files:
        print("No WAV files found in the given subfolder.")
        return

    for wav_file in wav_files:
        wav_path = os.path.join(subfolder, wav_file)

        # Load the audio file
        audio = AudioSegment.from_wav(wav_path)

        # Normalize, convert to mono, and set sample rate to 32 kHz
        audio = audio.normalize()
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(32000)

        # Save the modified audio back to the same file
        audio.export(wav_path, format="wav")

        print(f"Audio modified and saved for {wav_file}")


# Replace 'your_subfolder' with the actual path to your subfolder
normalize_audio_in_folder("dataset/dataset/")
