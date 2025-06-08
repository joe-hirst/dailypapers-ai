import logging
from pathlib import Path

import arxiv
from google.auth.credentials import Credentials
from google.oauth2.credentials import (
    Credentials as OAuth2Credentials,
)  # Renamed to avoid conflict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from src.settings import Settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def upload_video_to_youtube(
    video_path: Path,
    paper: arxiv.Result,
    settings: Settings,
) -> str:
    """Upload video to YouTube and return video ID."""
    logger.info("Starting YouTube upload for video: %s", video_path.name)
    paper_title = paper.title.strip()
    max_chars = 85
    if len(paper_title) >= max_chars:
        paper_title = f"{paper_title[:80]}..."
    title = f"{paper_title} (AI Podcast)".strip()

    description = f"""
Daily Papers podcast for {paper.published.date()}

Today's paper: {paper.title.strip()}
Paper URL: {paper.pdf_url.strip() if paper.pdf_url else "URL not availavle"}
Paper Authors: {", ".join([author.name for author in paper.authors])}

Daily Papers is an AI-generated podcast discussing the latest research papers in artificial intelligence and machine learning.

#AI #MachineLearning #Research #Podcast #DailyPapers
    """.strip()

    tags = [
        "AI",
        "Machine Learning",
        "Research",
        "Podcast",
        "Daily Papers",
        "Artificial Intelligence",
    ]
    category_id = 28  # Science & Technology

    try:
        # Get credentials
        creds = get_youtube_credentials(settings=settings)
        if not creds:
            msg = "Failed to get YouTube credentials"
            raise RuntimeError(msg)  # noqa: TRY301

        youtube = build("youtube", "v3", credentials=creds)

        # Video metadata
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags or [],
                "categoryId": category_id,
            },
            "status": {"privacyStatus": settings.youtube_video_privacy_status},
        }

        # Upload video
        media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)

        logger.info("Uploading video to YouTube...")
        request = youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media)

        # This is where the actual upload progress can be monitored if needed
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info("Uploaded %d%%.", int(status.progress() * 100))

        video_id = response["id"]

    except HttpError:
        logger.exception("YouTube API error during upload")
        raise
    except Exception:
        logger.exception("Unexpected error during YouTube upload")
        raise
    else:
        logger.info("Video uploaded successfully! Video ID: %s", video_id)
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info("Video URL: %s", video_url)
        return video_id


def get_youtube_credentials(settings: Settings) -> Credentials | None:
    """Get YouTube API credentials using a refresh token."""
    logger.info("Authenticating...")
    creds = None

    # Check if a token file exists (containing refresh token)
    creds = OAuth2Credentials(
        token=None,
        refresh_token=settings.youtube_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",  # noqa: S106
        client_id=settings.youtube_client_id,
        client_secret=settings.youtube_client_secret,
        scopes=SCOPES,
    )

    logger.info("Authenticated")
    return creds
