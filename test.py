from faster_whisper import WhisperModel
import pyaudio
import os
import wave

NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"


def transcribe_chunk(model, chunk_file):
    segments, info = model.transcribe(chunk_file, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def record_chunk(p, stream, chunk_file, chunk_length=1):
    # print("Recording...")
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    # print("Recording stopped")

    wf = wave.open(chunk_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    # Choose your model settings
    model_size = "large-v3"
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    accumulated_transcription = "" # Initialize an empty string to accumalate the transcription

    try:
        while True:
            chunk_file = "/home/david/Documents/hackathon2024/wamsley_to_douglascounty.mp3"
            record_chunk(p, stream, chunk_file)
            transcription = transcribe_chunk(model, chunk_file)
            print(NEON_GREEN + transcription + RESET_COLOR)
            os.remove(chunk_file)   

            # Append the new transcription to the accumulated transcription
            accumulated_transcription += transcription + " "        

    except KeyboardInterrupt:
        print("Stopping...")
        # Write the accumulated transcription to a file
        with open("transcription.txt", "w") as f:
            f.write(accumulated_transcription)

    finally:
        print("LOG: " + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()



if __name__ == "__main__":
    main()





# segments, info = model.transcribe("northhollywood_radio.mp3", beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))