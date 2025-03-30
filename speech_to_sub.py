import os
import sys
import pyaudio
import json
import numpy as np
import sqlite3
import time
from vosk import Model, KaldiRecognizer

# Database connection
conn = sqlite3.connect('engagement.db', check_same_thread=False)
c = conn.cursor()

MODEL_PATH = os.path.expanduser("~/Desktop/test/vosk-model")

if not os.path.exists(MODEL_PATH):
    print("Model not found! Make sure you downloaded the Vosk model correctly.")
    sys.exit(1)

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()

# (Optional) List available audio devices for debugging
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"Device {i}: {dev['name']}")

# Open microphone stream
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=4096
)

print("Listening... Speak into the microphone.")

def get_latest_speaker_id():
    """Fetch the most recent speaker_id from the database."""
    c.execute("SELECT speaker_id FROM engagement ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    return row[0] if row else None

def update_transcription(speaker_id, text):
    """Update the database with the transcribed text for the most recent speaker_id."""
    # (Optional) store a timestamp if you have a timestamp column:
    # timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    if speaker_id is None:
        speaker_id = 0  # Fallback if no speaker was found

    # Update transcription in the DB
    c.execute('''
        UPDATE engagement
        SET transcription = ?
        WHERE speaker_id = ? AND transcription IS NULL
    ''', (text, speaker_id))

    conn.commit()

try:
    while True:
        data = stream.read(4096, exception_on_overflow=False)

        # If Vosk detects the end of an utterance, we get a final result
        if recognizer.AcceptWaveform(data):
            final_result_json = recognizer.Result()
            final_result = json.loads(final_result_json)
            final_text = final_result.get("text", "")

            # Fetch the most recent speaker ID from the database
            speaker_id = get_latest_speaker_id()

            # Only now do we store the final text in the DB
            update_transcription(speaker_id, final_text)

            # Print the final recognized text
            print(f"Speaker {speaker_id if speaker_id else 'Unknown'} said: {final_text}")

        else:
            # Partial (intermediate) result for real-time display
            partial_result_json = recognizer.PartialResult()
            partial_result = json.loads(partial_result_json)
            partial_text = partial_result.get("partial", "")

            # Show partial text on the console but DO NOT store in the DB
            if partial_text:
                print(f"Partial: {partial_text}")

except KeyboardInterrupt:
    print("\nStopping transcription.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    conn.close()
