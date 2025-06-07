import logging
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def generate_script_from_paper(paper_path: Path, script_model: str, gemini_api_key: str) -> str | None:
    """Generates a podcast script from a research paper using a Gemini model."""
    logger.info("Starting script generation from paper: %s", paper_path.name)

    prompt = """
    Write a podcast script for this paper.
    The podcast should be around 8-10 minutes.
    The transcript should be in the following format, alternating between 'Speaker 1' and 'Speaker 2':
    Speaker 1: We're seeing a noticeable shift in consumer preferences across several sectors. What seems to be driving this change?
    Speaker 2: It appears to be a combination of factors, including greater awareness of sustainability issues and a growing demand for personalized experiences.
    Output transcript only.
    Ensure the podcast is fun and engaging for viewers.
    The name of the podcast is 'Daily Papers'.
    Make sure you explain all the concepts and make the podcast accessible to a wide audience.
    """

    client = genai.Client(api_key=gemini_api_key)

    try:
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=paper_path.read_bytes(), mime_type="application/pdf"),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        response = client.models.generate_content(
            model=script_model,
            contents=contents,
            config=generate_content_config,
        )

        podcast_script_raw = response.text
        if podcast_script_raw is None:
            logger.warning("Gemini model returned None for response text for paper: %s", paper_path.name)
            return None

        podcast_script = podcast_script_raw.strip()
        if not podcast_script:
            logger.warning("Gemini model returned an empty script for paper: %s", paper_path.name)
            return None

    except Exception:
        logger.exception("Failed to generate script for paper '%s'. An error occurred.", paper_path.name)
        return None
    else:
        logger.info("Script generation complete for paper: %s", paper_path.name)
        Path("data", "script.txt").write_text(podcast_script)
        return podcast_script
