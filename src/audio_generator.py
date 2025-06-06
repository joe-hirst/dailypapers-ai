import logging
import mimetypes
import subprocess
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def _save_binary_data(output_file_path: Path, data: bytes) -> None:
    """Saves binary data to a specified file."""
    logger.info("Saving binary data to: %s", output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_file_path.open("wb") as f:
            f.write(data)
    except OSError:
        logger.exception("Failed to save file %s.", output_file_path)
        raise

    logger.info("File saved successfully: %s", output_file_path)


def _convert_audio_with_ffmpeg(input_path: Path, output_path: Path, output_format: str) -> Path:
    """Converts an audio file to a specified format using FFmpeg."""
    logger.info("Converting %s to %s format...", input_path.name, output_format)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-acodec",
        output_format,
        str(output_path),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logger.exception("FFmpeg conversion failed: Command: '%s'", " ".join(e.cmd))
        error_msg = f"FFmpeg conversion failed: {e.stderr}"
        raise RuntimeError(error_msg) from e
    except FileNotFoundError as e:
        logger.exception("FFmpeg command not found. Is FFmpeg installed and in your PATH?")
        error_msg = "FFmpeg is not installed or not in system PATH."
        raise RuntimeError(error_msg) from e
    except Exception as e:
        logger.exception("An unexpected error occurred during FFmpeg conversion.")
        error_msg = f"An unexpected error occurred during FFmpeg conversion: {e}"
        raise RuntimeError(error_msg) from e
    else:
        logger.info("Successfully converted audio to: %s", output_path)
        return output_path


def _process_audio_stream(tts_model: str, contents: list[types.Content], config: types.GenerateContentConfig, gemini_api_key: str) -> tuple[bytes, str | None]:
    """Processes the audio stream from the Gemini API, accumulating data and MIME type."""
    full_audio_data = b""
    received_mime_type: str | None = None

    client = genai.Client(api_key=gemini_api_key)

    for chunk in client.models.generate_content_stream(model=tts_model, contents=contents, config=config):
        if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            if inline_data and inline_data.data:
                full_audio_data += inline_data.data
                if received_mime_type is None:
                    received_mime_type = inline_data.mime_type
            elif chunk.text:
                logger.info(chunk.text)
        else:
            logger.debug("Skipping empty or malformed chunk.")

    if not full_audio_data:
        logger.error("No audio data was received from the TTS model.")

    return full_audio_data, received_mime_type


def generate_audio_from_script(podcast_script: str, tts_model: str, gemini_api_key: str) -> Path:
    """Generates audio from a podcast script using the specified TTS model."""
    logger.info("Starting text-to-speech generation...")

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=podcast_script)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 1", voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda"))
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2", voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck"))
                    ),
                ],
            ),
        ),
    )

    data_dir = Path("data")
    temp_raw_audio_path = data_dir / "temp_raw_audio"
    final_wav_path = data_dir / "podcast.wav"

    try:
        full_audio_data, received_mime_type = _process_audio_stream(tts_model, contents, generate_content_config, gemini_api_key)

        if not full_audio_data:
            return Path()

        if received_mime_type:
            ext = mimetypes.guess_extension(received_mime_type)
            if ext:
                temp_raw_audio_path = temp_raw_audio_path.with_suffix(ext)
            else:
                logger.warning("Could not guess file extension for MIME type: %s. Saving as .bin.", received_mime_type)
                temp_raw_audio_path = temp_raw_audio_path.with_suffix(".bin")
        else:
            logger.warning("No MIME type provided for audio data. Saving as .bin.")
            temp_raw_audio_path = temp_raw_audio_path.with_suffix(".bin")

        _save_binary_data(temp_raw_audio_path, full_audio_data)

        if temp_raw_audio_path.suffix.lower() == ".wav":
            temp_raw_audio_path.rename(final_wav_path)
            logger.info("Audio was already WAV, moved to: %s", final_wav_path)
        else:
            _convert_audio_with_ffmpeg(temp_raw_audio_path, final_wav_path, "pcm_s16le")
            temp_raw_audio_path.unlink(missing_ok=True)

    except Exception:
        logger.exception("Error during audio generation pipeline.")
        return Path()
    else:
        return final_wav_path
