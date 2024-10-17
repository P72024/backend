from io import BytesIO
import re
from typing import List
from ASR.LocalAgreement import LocalAgreement
from faster_whisper import WhisperModel 

class ASR:
    audio_buffer: List[BytesIO] = []
    local_agreement = LocalAgreement()
    context:str = ""
    confirmed_sentences: List[str] = []
    def __init__ (self, model_size: str, device="auto", compute_type = "int8"):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        

    def transcribe(self, audio_buffer: BytesIO, context: str):
        transcribed_text = ""
        segments, info = self.whisper_model.transcribe(audio_buffer, beam_size=5, vad_filter=True, initial_prompt=context)
        #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        for segment in segments:
            #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            transcribed_text += " " + segment.text
            
        return transcribed_text



    
    def process_audio(self, audio_chunk: BytesIO) -> str:
        self.audio_buffer.append(audio_chunk)
        print(len(self.audio_buffer))
        audio_buffer_joined = BytesIO(b''.join(self.audio_buffer))
        transcribed_text = self.transcribe(audio_buffer_joined, self.context)
        print("transcribed_text: " + transcribed_text)
        confirmed_text = self.local_agreement.confirm_tokens(transcribed_text)
        print(confirmed_text)
        punctuation = r"[.!?]"  # Regular expression pattern for ., !, or ?
        # Detect punctuation
        print("check punctuation: ", re.search(punctuation,confirmed_text))
        if re.search(punctuation,confirmed_text):
            split_sentence = re.split(f"({punctuation})", confirmed_text)

            # Join the punctuation back to the respective parts of the sentence
            sentence = [split_sentence[i] + split_sentence[i+1] for i in range(0, len(split_sentence)-1, 2)]

            print("sentence", sentence)
            self.confirmed_sentences.append(sentence[-1])
            self.context = " ".join(self.confirmed_sentences)
            print("context added: " + self.context)
            
           
        
    
  
        
 

        