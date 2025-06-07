import logging
import struct
from contextlib import suppress
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def generate_audio_from_script(podcast_script: str, output_wav_path: Path, tts_model: str, gemini_api_key: str) -> None:
    """Generate audio from podcast script using text-to-speech model."""
    logger.info("Starting text-to-speech generation using model: %s", tts_model)

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
                        speaker="Speaker 1",
                        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda")),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2",
                        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")),
                    ),
                ],
            ),
        ),
    )

    try:
        full_audio_data, received_mime_type = _process_audio_stream(tts_model, contents, generate_content_config, gemini_api_key)

        if not full_audio_data:
            return

        if received_mime_type and not received_mime_type.lower().startswith("audio/wav"):
            full_audio_data = _convert_to_wav(full_audio_data, received_mime_type)
            logger.info("Converted received audio to WAV format.")
        elif not received_mime_type:
            logger.warning("No MIME type provided for audio data. Assuming default and attempting WAV conversion.")
            full_audio_data = _convert_to_wav(full_audio_data, "audio/L16;rate=24000")
    except Exception:
        logger.exception("Error during audio generation pipeline.")
    else:
        _save_binary_data(output_file_path=output_wav_path, data=full_audio_data)


def _process_audio_stream(tts_model: str, contents: list[types.Content], config: types.GenerateContentConfig, gemini_api_key: str) -> tuple[bytes, str | None]:
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


def _parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample: int = 16
    rate: int = 24000
    parts = mime_type.split(";")
    for part in parts:
        stripped_part = part.strip()
        if stripped_part.lower().startswith("rate="):
            with suppress(ValueError, IndexError):
                rate_str = stripped_part.split("=", 1)[1]
                rate = int(rate_str)
        elif stripped_part.startswith("audio/L"):
            with suppress(ValueError, IndexError):
                bits_per_sample = int(stripped_part.split("L", 1)[1])
    return {"bits_per_sample": bits_per_sample, "rate": rate}


def _convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = _parse_audio_mime_type(mime_type)
    bits_per_sample: int = parameters.get("bits_per_sample", 16)
    sample_rate: int = parameters.get("rate", 24000)

    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def _save_binary_data(output_file_path: Path, data: bytes) -> None:
    logger.info("Saving binary data to: %s", output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_file_path.open("wb") as f:
            f.write(data)
    except OSError:
        logger.exception("Failed to save file %s.", output_file_path)
        raise
    logger.info("File saved successfully: %s", output_file_path)
