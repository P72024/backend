# import os

from faster_whisper import WhisperModel

model_size = "tiny"

# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# or run on CPU with INT8
model = WhisperModel(model_size, device="auto", compute_type="int8")

def transcribe(audio_file):
    segments, info = model.transcribe(audio_file, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    if(info.language):
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
