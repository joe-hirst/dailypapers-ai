import logging
import re
from datetime import date
from pathlib import Path

import arxiv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def find_and_download_paper(date: date, output_paper_path: Path, paper_selector_model: str, gemini_api_key: str) -> arxiv.Result | None:
    """Find best paper for date and download it to specified path."""
    logger.info("Starting paper selection and download process for %s", date.isoformat())

    # 1. Get list of abstracts uploaded to Arxiv
    papers_with_abstracts = get_abstracts_for_day(date)
    if not papers_with_abstracts:
        logger.warning("No papers found for date %s, skipping paper selection", date.isoformat())
        return None

    # 2. Select the best paper
    best_paper = select_paper_for_podcast(papers_with_abstracts=papers_with_abstracts, paper_selector_model=paper_selector_model, gemini_api_key=gemini_api_key)
    if not best_paper:
        logger.error("Failed to select best paper from available candidates")
        return None

    # 3. Download the best paper
    return download_arxiv_pdf_from_url(pdf_url=best_paper, output_paper_path=output_paper_path)


def get_abstracts_for_day(target_date: date, max_results: int = 500) -> list[str] | None:
    """Fetch list of AI papers from arXiv for specific date."""
    logger.info("Fetching AI papers for date: %s (max: %d)", target_date.isoformat(), max_results)
    client = arxiv.Client()

    # Query for AI papers (cat:cs.AI) submitted on the target date
    search_query = f"cat:cs.AI AND submittedDate:[{target_date.strftime('%Y%m%d')}000000 TO {target_date.strftime('%Y%m%d')}235959]"

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = list(client.results(search))
    if not results:
        logger.warning("No AI papers found for %s", target_date.isoformat())
        return None

    logger.info("Found %d AI papers for %s", len(results), target_date.isoformat())  # Add this
    return [
        f"""
        Title: {paper.title}
        Authors: {", ".join([author.name for author in paper.authors])}
        PDF URL: {paper.pdf_url}
        Abstract: {paper.summary}{paper.download_pdf}
        ----------------------------------------
        """
        for paper in results
    ]


def select_paper_for_podcast(papers_with_abstracts: list[str], paper_selector_model: str, gemini_api_key: str) -> str | None:
    """Select best paper for podcast from list of abstracts using LLM."""
    logger.info("Selecting best paper from %d candidates using model: %s", len(papers_with_abstracts), paper_selector_model)

    prompt = f"""
    Take a look at the following papers. Select the ""best"" one to be discussed on the Daily Papers podcast.
    Return the pdf_url only.

    <papers>
    {"".join(papers_with_abstracts)}
    </papers>
    """

    client = genai.Client(api_key=gemini_api_key)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    response = client.models.generate_content(
        model=paper_selector_model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text


def download_arxiv_pdf_from_url(pdf_url: str, output_paper_path: Path) -> arxiv.Result | None:
    """Download arXiv paper PDF from URL to specified path."""
    logger.info("Attempting to download paper from URL: %s", pdf_url)

    # 1. Extract the arXiv ID from the PDF URL
    match = re.search(r"arxiv\.org/pdf/(\d{4}\.\d{5}(?:v\d+)?)", pdf_url)
    if not match:
        logger.warning("Could not extract arXiv ID from URL: %s", pdf_url)
        return None

    arxiv_id = match.group(1)
    logger.info("Extracted arXiv ID: %s", arxiv_id)

    # 2. Use the arxiv SDK to search for that ID
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search), None)
        if paper:
            filepath = paper.download_pdf(dirpath=str(output_paper_path.parent), filename=output_paper_path.name)
            logger.info("Downloaded %s to %s", paper.title, filepath)
            logger.info("Paper was submitted on %s", paper.published.date())
            return paper
        logger.error("No paper found for arXiv ID: %s", arxiv_id)

    except Exception:
        logger.exception("Failed to download paper from arXiv ID: %s", arxiv_id)
