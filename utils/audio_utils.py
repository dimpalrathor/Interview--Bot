import speech_recognition as sr
import whisper
import tempfile
import os
import numpy as np
import soundfile as sf

# Optional: try to import noisereduce, but keep functionality even if it's missing.
try:
    import noisereduce as nr
    _have_noisereduce = True
except Exception:
    _have_noisereduce = False

class AudioHandler:
    def __init__(self, device_index=1, model_size="small"):
        print("🎙 Loading Whisper model:", model_size)
        try:
            self.model = whisper.load_model(model_size)
        except Exception as e:
            print("❌ Could not load requested whisper model, falling back to 'tiny':", e)
            self.model = whisper.load_model("tiny")
        self.recognizer = sr.Recognizer()
        self.device_index = device_index

    def listen(self, timeout=12, phrase_time_limit=30, calibrate_seconds=1.2):
        """
        Record audio, apply optional noise reduction, and transcribe with Whisper.
        Longer phrase_time_limit so user answers aren't cut.
        """
        try:
            with sr.Microphone(device_index=self.device_index) as source:
                print("🎤 Listening...")
                # longer ambient calibration
                self.recognizer.adjust_for_ambient_noise(source, duration=calibrate_seconds)
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
        except Exception as e:
            print("❌ Listen error:", e)
            return ""

        # Save WAV temp
        wav_path = None
        reduced_wav = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.get_wav_data())
                wav_path = tmp.name

            # Load wav with soundfile
            audio_data, sr_ = sf.read(wav_path)

            # If stereo, convert to mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Optional noise reduction
            if _have_noisereduce:
                try:
                    reduced = nr.reduce_noise(y=audio_data.astype(np.float32), sr=sr_)
                except Exception as e:
                    print("⚠️ noisereduce failed:", e)
                    reduced = audio_data
            else:
                reduced = audio_data

            # write reduced to a new temp file for whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
                sf.write(tmp2.name, reduced, sr_)
                reduced_wav = tmp2.name

            # Whisper transcription (use language='en' to prefer English)
            print("🧠 Whisper transcribing…")
            try:
                result = self.model.transcribe(reduced_wav, language='en', temperature=0.0)
                text = result.get('text', '').strip().lower()
            except Exception as e:
                print("❌ Whisper transcribe failed:", e)
                text = ""

            # cleanup temp files
            try:
                os.remove(wav_path)
            except Exception:
                pass
            try:
                if reduced_wav and os.path.exists(reduced_wav):
                    os.remove(reduced_wav)
            except Exception:
                pass

            return text

        except Exception as e:
            print("❌ Whisper error:", e)
            # cleanup
            if wav_path and os.path.exists(wav_path):
                try: os.remove(wav_path)
                except Exception: pass
            if reduced_wav and os.path.exists(reduced_wav):
                try: os.remove(reduced_wav)
                except Exception: pass
            return ""

    def speak(self, text):
        """Simple TTS via pyttsx3"""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)
        engine.say(text)
        engine.runAndWait()
