import logging
from datetime import date
from pathlib import Path

import arxiv
import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Paper(BaseModel):
    title: str
    reason_for_choice: str
    arxiv_id: str


def find_and_download_paper(target_date: date, output_paper_path: Path, paper_selector_model: str, gemini_api_key: str) -> arxiv.Result:
    """Find best paper for date and download it to specified path."""
    logger.info("Starting paper selection and download process for %s", target_date.isoformat())

    # 1. Get list of abstracts uploaded to Arxiv
    papers_with_abstracts = get_abstracts_for_day(target_date)

    # 2. Select the best paper
    best_paper_id = select_paper_for_podcast(papers_with_abstracts=papers_with_abstracts, paper_selector_model=paper_selector_model, gemini_api_key=gemini_api_key)

    # 4. Get arXiv object
    best_paper_result = get_arxiv_paper(best_paper_id)

    # 3. Download the best paper
    if not best_paper_result.pdf_url:
        msg = "Selected paper has no PDF URL available"
        raise ValueError(msg)
    download_file(url=best_paper_result.pdf_url, output_path=output_paper_path)

    return best_paper_result


def get_abstracts_for_day(target_date: date, max_results: int = 500) -> list[str]:
    """Fetch list of AI papers from arXiv for specific date."""
    logger.info("Fetching AI papers for date: %s (max: %d)", target_date.isoformat(), max_results)
    client = arxiv.Client()

    # Query for AI papers (cat:cs.AI) submitted on the target date
    primary_ai_categories = ["cs.AI", "cs.LG"]
    category_query = " OR ".join([f"cat:{cat}" for cat in primary_ai_categories])
    search_query = f"({category_query}) AND submittedDate:[{target_date.strftime('%Y%m%d')}000000 TO {target_date.strftime('%Y%m%d')}235959]"
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = list(client.results(search))
    if not results:
        msg = f"No AI papers found for %s {target_date.isoformat()}"
        raise ValueError(msg)

    logger.info("Found %d AI papers for %s", len(results), target_date.isoformat())  # Add this
    return [
        f"""
        Title: {paper.title}
        Authors: {", ".join([author.name for author in paper.authors])}
        arXiv ID: {paper.get_short_id()}
        Summary: {paper.summary}
        ----------------------------------------
        """
        for paper in results
    ]


def select_paper_for_podcast(papers_with_abstracts: list[str], paper_selector_model: str, gemini_api_key: str) -> str:
    """Select best paper for podcast from list of abstracts using LLM."""
    logger.info("Selecting best paper from %d candidates using model: %s", len(papers_with_abstracts), paper_selector_model)

    prompt = f"""
    Select the most impactful paper to be discussed on the Daily Papers podcast.
    Pick a paper based on the following criteria:
    - Interesting results in Artificial Intelligence
    - Broad appeal to a large audience
    - Potential for virality
    - Credible and established authors/institutions

    <papers>
    {"".join(papers_with_abstracts)}
    </papers>
    """
    logger.info("Papers with summaries have %s words", len(prompt.split()))

    client = genai.Client(api_key=gemini_api_key)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Paper,
    )
    response = client.models.generate_content(
        model=paper_selector_model,
        contents=contents,
        config=generate_content_config,
    )
    result = response.parsed
    if not isinstance(result, Paper):
        msg = "LLM returned invalid response"
        raise TypeError(msg)

    return result.arxiv_id


def download_file(url: str, output_path: Path) -> None:
    """Download file from URL to specified path."""
    response = httpx.get(url)
    response.raise_for_status()
    output_path.write_bytes(response.content)

    # Validate PDF
    if not output_path.read_bytes()[:4].startswith(b"%PDF"):
        output_path.unlink()
        msg = "Downloaded file is not a valid PDF"
        raise ValueError(msg)


def get_arxiv_paper(arxiv_id: str) -> arxiv.Result:
    """Get arxiv paper object from ID."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)
    if not paper:
        msg = f"Could not find paper with arXiv ID: {arxiv_id}"
        raise ValueError(msg)
    return paper
