import librosa
import numpy as np
import jiwer
from src.ASR.ASR import ASR
async def process_audio_benchmark(wav_filename, txt_filename):
    print(f"HERE {wav_filename}!!!!!")
    try:
        print(f"Processing {wav_filename} and {txt_filename}")
        transcribed_text = ""
        actual_text = ""
        asr = ASR(model_size="base", device="auto", compute_type="float32")
        audio, _ = librosa.load(wav_filename, sr=16000, dtype=np.float32)
        # Chunk the audio into 2 second chunks
        chunk_size = 16000 * 2
        for i in range(0, len(audio), chunk_size):
            transcribed_text += asr.process_audio(audio[i:i + chunk_size])
        with open(txt_filename, "r") as f:
            actual_text = f.read()
        transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        wer = jiwer.wer(actual_text, transcribed_text, truth_transform=transforms, hypothesis_transform=transforms)
        print(f"WER: {wer} between {txt_filename} and {wav_filename}")
    except Exception as e:
        print(f"Error processing {wav_filename} and {txt_filename}: {e}")