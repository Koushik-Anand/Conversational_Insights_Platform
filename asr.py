import speech_recognition as sr
from pydub import AudioSegment

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    audio_format = file_path.split('.')[-1]
    
    # Convert to WAV if necessary
    if audio_format != 'wav':
        audio = AudioSegment.from_file(file_path)
        file_path = file_path.replace(audio_format, 'wav')
        audio.export(file_path, format='wav')
    
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Error connecting to recognition service"
