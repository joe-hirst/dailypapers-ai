from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", alias="LOG_LEVEL")
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")
    script_model: str = Field(alias="GEMINI_SCRIPT_MODEL")
    tts_model: str = Field(alias="GEMINI_TTS_MODEL")
    paper_date: str = Field(default_factory=lambda: (datetime.now(tz=UTC) - timedelta(days=3)).strftime("%Y-%m-%d"), alias="PAPER_DATE")

    @property
    def paper_date_parsed(self) -> date:
        """Parse paper_date string into date object."""
        return date.fromisoformat(self.paper_date)

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance."""
    return Settings()  # pyright: ignore[reportCallIssue]
