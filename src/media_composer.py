import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def compose_final_podcast_video(audio_wav_path: Path, background_image: Path) -> None:
    mp3_path = convert_wav_to_mp3(input_wav_path=audio_wav_path)
    create_video_from_mp3_and_image(input_mp3_path=mp3_path, background_image=background_image)


def convert_wav_to_mp3(input_wav_path: Path) -> Path:
    logger.info("Converting wav to mp3")
    output_mp3_path = input_wav_path.with_suffix(".mp3")
    command = [
        "ffmpeg",
        "-y",  # Enable overwriting
        "-i",
        str(input_wav_path),  # Input file
        "-vn",  # No video recording (audio-only)
        "-acodec",
        "libmp3lame",  # Use libmp3lame for MP3 encoding
        "-q:a",
        "2",  # Quality (0-9, 0 is best, 2 is high quality)
        str(output_mp3_path),  # Output file
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return output_mp3_path


def create_video_from_mp3_and_image(input_mp3_path: Path, background_image: Path) -> None:
    logger.info("Creating mp4 file")
    output_video_path = Path("data", "podcast.mp4")
    command = [
        "ffmpeg",
        "-y",  # Enable overwriting
        "-loop",
        "1",  # Loop the image
        "-i",
        str(background_image),  # Input image
        "-i",
        str(input_mp3_path),  # Input MP3
        "-c:v",
        "libx264",  # Video codec
        "-c:a",
        "copy",  # Copy the audio stream directly (no re-encoding)
        "-shortest",  # Finish when the shortest input stream finishes (MP3)
        "-pix_fmt",
        "yuv420p",  # Pixel format (required for many players)
        str(output_video_path),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
