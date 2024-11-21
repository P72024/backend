import logging
from io import BytesIO
from unittest.mock import patch

from pydub import AudioSegment
from pydub.generators import Sine

from src.ASR.ASR import ASR


class TestASR():
    _ASR = ASR("tiny", device="auto", compute_type="int8", max_context_length=100, chunk_limit=3)

    def test_save_metadata(self):
        """
        Test that metadata is properly saved.
        """
        # Make sure its empty at initialization
        assert self._ASR.metadata.getvalue() == b''

        # Create some random bytes
        rand_bytes = b'abcxyz123'

        # Save the bytes as metadata
        self._ASR.save_metadata(rand_bytes)

        # Assert the metadata was correctly set
        assert self._ASR.metadata.getvalue() == b'abcxyz123'

    def test_receive_audio_chunk(self):
        """
        Test that receive_audio_chunk correctly appends bytes to the audio_buffer.
        Test that receive_audio_chunk calls process_audio when the audio_buffer is large enough.
        """
        with patch.object(self._ASR, "process_audio", return_value="processed!") as mock_process_audio:
            for i in range(1,4):
                assert len(self._ASR.audio_buffer) == i-1
                assert self._ASR.receive_audio_chunk(b'testbyte') == ""
                assert len(self._ASR.audio_buffer) == i
            mock_process_audio.assert_not_called()
            assert self._ASR.receive_audio_chunk(b'test') == "processed!"
            mock_process_audio.assert_called_once()

    def test_process_audio(self):
        """
        Test that the process_audio function correctly returns a formatted transcription that doesnt contain leading '...' or '-'.
        Test that the process_audio function correctly clears the audio_buffer if the audio coming in is silent.
        Test that the context gets updated with the new transcription
        Test that the audio_buffer is after updating the context
        
        mocked:
            - whisper transcription is mocked to something we can expect.
            - is_silent is mocked to return true or false depending on what we test. 
        """

        # is_silent = True
        with patch.object(self._ASR, "transcribe", return_value="hello world!") as mock_transcribe:
            with patch.object(self._ASR, "is_silent", return_value=True) as mock_is_silent:
                self._ASR.audio_buffer = []
                mockbyte = b'someaudiobyte'
                mockbyte2 = b'someaudiobyte'
                self._ASR.audio_buffer.append(BytesIO(mockbyte))
                self._ASR.audio_buffer.append(BytesIO(mockbyte2))
                assert len(self._ASR.audio_buffer) == 2
                assert self._ASR.process_audio() == ""
                assert len(self._ASR.audio_buffer) == 0

        # is_silent = False
        with patch.object(self._ASR, "transcribe", return_value="hello world!") as mock_transcribe:
            with patch.object(self._ASR, "is_silent", return_value=False) as mock_is_silent:
                # Create a mock audio_buffer
                self._ASR.audio_buffer = []
                mockbyte = b'someaudiobyte'
                mockbyte2 = b'someaudiobyte'
                self._ASR.audio_buffer.append(BytesIO(mockbyte))
                self._ASR.audio_buffer.append(BytesIO(mockbyte2))
                assert len(self._ASR.audio_buffer) == 2
                assert self._ASR.context == ''
                assert self._ASR.process_audio() == "hello world!"
                assert self._ASR.context == "hello world!"
                assert len(self._ASR.audio_buffer) == 0

                # Test that the context is appended on new runs
                self._ASR.audio_buffer = []
                mockbyte = b'someaudiobyte'
                mockbyte2 = b'someaudiobyte'
                self._ASR.audio_buffer.append(BytesIO(mockbyte))
                self._ASR.audio_buffer.append(BytesIO(mockbyte2))
                self._ASR.process_audio()
                assert self._ASR.context == "hello world! hello world!"

    def test_update_context(self):
        """
        Test that the context is correctly updated
        Test that the context is correctly split when the context is getting too big.
        """
        self._ASR.context = ""
        self._ASR.max_context_length = 10
        self._ASR.update_context("this is a test string")
        assert self._ASR.context == "this is a test string"
        self._ASR.update_context("and this is another") 
        assert self._ASR.context == "this is a test string and this is another"
        self._ASR.update_context("update")
        assert self._ASR.context == "another update"

    def test_is_silent(self):
        """
        Test that is_silent returns true if the audio_bytes coming in is too silent (has an RMS less than the threshold)
        Test that is_silent returns false if the audio_bytes coming in is not silent (has an RMS over the threshold)
        """
        silent_audio = AudioSegment.silent(duration=500)  # 500ms = 0.5 seconds
        mock_buffer = BytesIO()
        silent_audio.export(mock_buffer, format="webm", codec="libopus")
        assert True == self._ASR.is_silent(mock_buffer)

        non_silent_audio = Sine(440).to_audio_segment(duration=500)  # 440 Hz tone
        mock_buffer = BytesIO()
        non_silent_audio.export(mock_buffer, format="webm", codec="libopus")
        assert False == self._ASR.is_silent(mock_buffer)
