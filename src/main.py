import logging

from src.audio_generator import generate_audio_from_script
from src.script_generator import generate_script_from_paper
from src.settings import Settings, get_settings

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def podcast_generation_pipeline(settings: Settings) -> None:
    generate_script_from_paper(settings=settings)
    generate_audio_from_script(settings=settings)


if __name__ == "__main__":
    podcast_generation_pipeline(settings=settings)
