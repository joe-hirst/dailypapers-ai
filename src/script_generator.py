import logging
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def generate_script_from_paper(paper_path: Path, script_generator_model: str, gemini_api_key: str) -> str:
    """Generate podcast script from a research paper using a Gemini model."""
    logger.info("Starting script generation from paper: %s using model: %s", paper_path.name, script_generator_model)

    prompt = """
    Write a 6-10 minute podcast script for this paper.
    The name of the podcast is 'Daily Papers'.

    Format: Alternating dialogue between speakers labelled as 'Speaker 1' and 'Speaker 2' only.

    Requirements:
    - Accessible to a wide general audience
    - Explain all concepts and define technical terms simply
    - Ensure the podcast is fun and engaging for listeners
    - Friendly, conversational tone
    - Do not give the two speakers names and they should not refer to each other by name
    - The script should include speach from the two speakers only, do not include any stage directions
    - Script should include "Welcome to Daily Papers"

    Output: Transcript only.
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
            model=script_generator_model,
            contents=contents,
            config=generate_content_config,
        )

        podcast_script_raw = response.text
        if podcast_script_raw is None:
            msg = f"Gemini model returned None for response text for paper: {paper_path.name}"
            logger.warning(msg)
            raise RuntimeError(msg)  # noqa: TRY301

        podcast_script = podcast_script_raw.strip()
        if not podcast_script:
            msg = f"Gemini model returned an empty script for paper: {paper_path.name}"
            logger.warning(msg)
            raise RuntimeError(msg)  # noqa: TRY301

    except Exception:
        logger.exception("Failed to generate script for paper '%s'", paper_path.name)
        raise
    else:
        logger.info("Script generation complete for paper: %s (%d characters)", paper_path.name, len(podcast_script))
        return podcast_script
