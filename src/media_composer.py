import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def compose_final_podcast_video(input_wav_path: Path, output_mp4_path: Path, background_image: Path) -> None:
    """Composes the final podcast video from an audio file and a background image."""
    output_mp3_path = input_wav_path.with_suffix(".mp3")
    mp3_path = convert_wav_to_mp3(input_wav_path=input_wav_path, output_mp3_path=output_mp3_path)
    create_video_from_mp3_and_image(input_mp3_path=mp3_path, output_mp4_path=output_mp4_path, background_image=background_image)


def convert_wav_to_mp3(input_wav_path: Path, output_mp3_path: Path) -> Path:
    """Converts a WAV audio file to MP3 format using FFmpeg."""
    logger.info("Converting wav to mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_mp3_path),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logger.exception("FFmpeg WAV to MP3 conversion failed: Command: '%s'", " ".join(e.cmd))
        error_msg = f"FFmpeg WAV to MP3 conversion failed: {e.stderr}"
        raise RuntimeError(error_msg) from e
    except FileNotFoundError as e:
        logger.exception("FFmpeg command not found. Is FFmpeg installed and in your PATH?")
        error_msg = "FFmpeg is not installed or not in system PATH."
        raise RuntimeError(error_msg) from e
    except Exception as e:
        logger.exception("An unexpected error occurred during FFmpeg WAV to MP3 conversion.")
        error_msg = f"An unexpected error occurred during FFmpeg WAV to MP3 conversion: {e}"
        raise RuntimeError(error_msg) from e
    else:
        logger.info("Successfully converted WAV to MP3: %s", output_mp3_path)
        return output_mp3_path


def create_video_from_mp3_and_image(input_mp3_path: Path, output_mp4_path: Path, background_image: Path) -> None:
    """Creates an MP4 video from an MP3 audio file and a static background image using FFmpeg."""
    logger.info("Creating mp4 file")
    command = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(background_image),
        "-i",
        str(input_mp3_path),
        "-c:v",
        "libx264",
        "-c:a",
        "copy",
        "-shortest",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4_path),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logger.exception("FFmpeg MP4 creation failed: Command: '%s'", " ".join(e.cmd))
        error_msg = f"FFmpeg MP4 creation failed: {e.stderr}"
        raise RuntimeError(error_msg) from e
    except FileNotFoundError as e:
        logger.exception("FFmpeg command not found. Is FFmpeg installed and in your PATH?")
        error_msg = "FFmpeg is not installed or not in system PATH."
        raise RuntimeError(error_msg) from e
    except Exception as e:
        logger.exception("An unexpected error occurred during FFmpeg MP4 creation.")
        error_msg = f"An unexpected error occurred during FFmpeg MP4 creation: {e}"
        raise RuntimeError(error_msg) from e
    else:
        logger.info("Successfully created MP4 video: %s", output_mp4_path)
