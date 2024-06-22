import whisper
import pandas as pd
import os

# Load metadata
metadata = pd.read_excel('/home/david/Documents/hackathon2024/Challenge5-call_001-100/911_metadata.xlsx')

# Initialize Whisper model
model = whisper.load_model("base")

# Function to transcribe audio
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result['text']

# Add transcription column to metadata
metadata['transcription'] = metadata['filename'].apply(lambda x: transcribe_audio(f"data/train/{x}"))

# Save metadata with transcriptions
metadata.to_csv('/home/david/Documents/hackathon2024/data/metadata_with_transcriptions.csv', index=False)
