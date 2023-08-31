from pydub import AudioSegment
import os


def convert_mp3_to_wav(subfolder):
    # Get the list of all files in the subfolder
    files = [
        f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))
    ]

    # Filter out the files that are not MP3s
    mp3_files = [f for f in files if f.lower().endswith(".mp3")]

    # If no MP3 files are found, exit the function
    if not mp3_files:
        print("No MP3 files found in the given subfolder.")
        return

    # Loop through the MP3 files and convert them to WAV
    for mp3_file in mp3_files:
        mp3_path = os.path.join(subfolder, mp3_file)
        wav_path = os.path.join(subfolder, mp3_file.replace(".mp3", ".wav"))

        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")

        print(f"Converted {mp3_file} to WAV format.")


# Replace 'your_subfolder' with the actual path to your subfolder
convert_mp3_to_wav("dataset/dataset/")
