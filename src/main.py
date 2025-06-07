import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.audio_generator import generate_audio_from_script
from src.media_composer import compose_final_podcast_video
from src.paper_selector import find_and_download_paper
from src.script_generator import generate_script_from_paper
from src.settings import Settings, get_settings

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def podcast_generation_pipeline(settings: Settings) -> None:
    """Orchestrates the full podcast generation pipeline."""
    logger.info("Starting podcast generation pipeline.")

    background_image_path = Path("assets", "background.png")
    if not background_image_path.exists():
        logger.critical("Background image not found at %s. Ensure the path is correct and the file exists.", background_image_path)
        sys.exit(1)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Select best paper for day
        yesterday = datetime.now(tz=UTC) - timedelta(days=2)
        paper_path = data_dir / "paper.pdf"
        find_and_download_paper(date=yesterday, output_paper_path=paper_path, gemini_model=settings.script_model, gemini_api_key=settings.gemini_api_key)

        if not paper_path.exists():
            logger.critical("Paper not found at %s. Ensure the path is correct and the file exists.", paper_path)
            sys.exit(1)

        # 2. Generate podcast script
        logger.info("Generating podcast script from paper: %s", paper_path)
        podcast_script = generate_script_from_paper(paper_path=paper_path, script_model=settings.script_model, gemini_api_key=settings.gemini_api_key)

        if not podcast_script:
            logger.critical("Failed to generate podcast script. Script content was empty.")
            sys.exit(1)
        # Save script for reference
        script_path = data_dir / "podcast_script.txt"
        script_path.write_text(podcast_script)
        logger.info("Podcast script generated successfully.")

        # 3. Generate audio from the script
        logger.info("Generating audio from the podcast script.")
        audio_wav_path = data_dir / "podcast.wav"
        generate_audio_from_script(
            podcast_script=podcast_script, output_wav_path=audio_wav_path, tts_model=settings.tts_model, gemini_api_key=settings.gemini_api_key
        )

        if not audio_wav_path.exists():
            logger.critical("Audio generation failed: Output file does not exist.")
            sys.exit(1)
        logger.info("Audio generated successfully: %s", audio_wav_path)

        # 4. Compose the final video
        logger.info("Composing the final podcast video.")
        video_mp4_path = data_dir / "podcast.mp4"
        compose_final_podcast_video(input_wav_path=audio_wav_path, output_mp4_path=video_mp4_path, background_image=background_image_path)
        logger.info("Podcast video composed successfully.")

    except Exception:
        logger.exception("An unexpected error occurred during the podcast generation pipeline")
        sys.exit(1)
    else:
        logger.info("Podcast generation pipeline completed successfully.")


if __name__ == "__main__":
    podcast_generation_pipeline(settings=settings)
