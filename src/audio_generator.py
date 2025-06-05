import logging
import mimetypes
import struct
from contextlib import suppress
from pathlib import Path

from google import genai
from google.genai import types

from src.settings import Settings

logger = logging.getLogger(__name__)


def save_binary_file(file_name: str, data: bytes) -> None:
    """Saves binary data to a specified file."""
    logger.info("Saving to disk")
    with Path("data", file_name).open("wb") as f:
        f.write(data)
    logger.info("File saved to: %s", file_name)


def generate_audio_from_script(settings: Settings) -> None:
    """Generates audio from a transcript using a given audio model."""
    script = Path("data", "transcript.txt").read_text()
    logger.info("Generating TTS")
    client = genai.Client(api_key=settings.gemini_api_key)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=script),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 1",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Leda",
                            ),
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Puck",
                            ),
                        ),
                    ),
                ],
            ),
        ),
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=settings.tts_model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.candidates is None or chunk.candidates[0].content is None or chunk.candidates[0].content.parts is None:
            continue
        inline_data = chunk.candidates[0].content.parts[0].inline_data
        if inline_data and inline_data.data:
            file_name_prefix = f"podcast_{file_index}"
            file_index += 1

            file_extension = mimetypes.guess_extension(inline_data.mime_type or "")
            data_buffer = inline_data.data

            if file_extension is None:
                file_extension = ".wav"
                if inline_data.mime_type:
                    data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
                else:
                    logger.warning("MIME type is missing for audio data, cannot convert to WAV without it.")
                    continue
            save_binary_file(f"{file_name_prefix}{file_extension}", data_buffer)
        elif chunk.text:
            logger.info(chunk.text)


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters."""
    logger.info("Converting to wav")
    parameters = parse_audio_mime_type(mime_type)

    bits_per_sample = parameters.get("bits_per_sample", 16)
    sample_rate = parameters.get("rate", 24000)

    num_channels = 1
    data_size = len(audio_data)

    if bits_per_sample is None:
        logger.error("bits_per_sample is None, cannot calculate bytes_per_sample.")
        return b""

    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample

    if sample_rate is None:
        logger.error("sample_rate is None, cannot calculate byte_rate.")
        return b""

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


def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string."""
    logger.info("Parsing audio MIME type")
    bits_per_sample: int | None = 16
    rate: int | None = 24000

    parts = mime_type.split(";")
    for original_param_str in parts:
        param_stripped = original_param_str.strip()
        if param_stripped.lower().startswith("rate="):
            with suppress(ValueError, IndexError):
                rate = int(param_stripped.split("=", 1)[1])
        elif param_stripped.startswith("audio/L"):
            with suppress(ValueError, IndexError):
                bits_per_sample = int(param_stripped.split("L", 1)[1])

    return {"bits_per_sample": bits_per_sample, "rate": rate}
