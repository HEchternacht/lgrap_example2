"""
Application configuration via pydantic-settings.
Values can be set via environment variables or a .env file.
LangSmith tracing is automatically configured on import.
"""
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    openai_base_url: str = "http://127.0.0.1:8080/v1"
    openai_api_key: str = "not-needed"
    model_name: str = "gpt-4"

    # --- LangSmith monitoring (new-style LANGSMITH_ prefix) ---
    langsmith_tracing: str = "false"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "lgrap"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    def configure_langsmith(self) -> None:
        """Export LangSmith env vars (both new LANGSMITH_ and legacy LANGCHAIN_ prefixes)."""
        # New-style — picked up by langsmith SDK >= 0.2
        os.environ["LANGSMITH_TRACING"] = self.langsmith_tracing
        os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        # Legacy aliases — still required by some LangChain internals
        os.environ["LANGCHAIN_TRACING_V2"] = self.langsmith_tracing
        os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        if self.langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key


# Singleton — imported by all modules
settings = Settings()
settings.configure_langsmith()
