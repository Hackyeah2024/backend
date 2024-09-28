import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip


def transcribe(audio_path):
    try:
        # Load the Whisper model
        model = whisper.load_model("base")

        # Transcribe the audio
        result = model.transcribe(audio_path, word_timestamps= True)

        segments = [
            {"text": seg["text"].strip(), "from": seg["start"], "to": seg["end"]}
            for seg in result["segments"]
        ]
        return result["text"], segments

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        transcription = "An error occurred during transcription."

    return transcription, []


def extract_audio_file(audio_path, video_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)
