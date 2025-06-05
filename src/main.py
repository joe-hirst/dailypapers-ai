import logging
from pathlib import Path

from src.audio_generator import generate_audio_from_script
from src.media_composer import compose_final_podcast_video
from src.script_generator import generate_script_from_paper
from src.settings import Settings, get_settings

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

paper_path = "data/papers/2506.03095v1.pdf"


def podcast_generation_pipeline(settings: Settings) -> None:
    podcast_script = generate_script_from_paper(paper_path=paper_path, script_model=settings.script_model, gemini_api_key=settings.gemini_api_key)
    if not podcast_script:
        return
    audio_wav_path = generate_audio_from_script(podcast_script=podcast_script, tts_model=settings.tts_model, gemini_api_key=settings.gemini_api_key)
    compose_final_podcast_video(audio_wav_path=audio_wav_path, background_image=Path("assets", "background.png"))


if __name__ == "__main__":
    podcast_generation_pipeline(settings=settings)
