import logging

from src.make_audio import make_audio
from src.make_script import make_script

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Making script")
    make_script(script_model="gemini-2.5-pro-preview-05-06")
    logger.info("Making audio")
    make_audio(audio_model="gemini-2.5-pro-preview-tts")


if __name__ == "__main__":
    main()
