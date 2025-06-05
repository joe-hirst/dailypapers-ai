import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)


def generate_script_from_paper(script_model: str = "gemini-2.5-pro-preview-05-06") -> None:
    logger.info("Making script")
    paper_path = Path("data", "papers", "2506.03095v1.pdf")
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

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=paper_path.read_bytes(), mime_type="application/pdf"),
            ],
        ),
    ]
    tools = [types.Tool(url_context=types.UrlContext())]
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )
    response = client.models.generate_content(
        model=script_model,
        contents=contents,
        config=generate_content_config,
    )

    logger.info("Saving transcript")
    if response.text is not None:
        Path("data", "transcript.txt").write_text(response.text)


if __name__ == "__main__":
    generate_script_from_paper()
