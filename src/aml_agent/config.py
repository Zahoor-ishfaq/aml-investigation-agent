"""
Application configuration.

Loads and validates environment variables at import time. Using
pydantic-settings (not raw os.getenv) means a missing or malformed
required variable raises immediately at startup — failing here is
far cheaper than failing inside a DB call three modules deep.

Reference: pydantic-settings docs, https://docs.pydantic.dev/latest/concepts/pydantic_settings/
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed application settings, sourced from environment variables / .env file.

    Individual postgres_* fields are the source of truth (not a raw DSN string
    in .env) so credentials can be validated/typed individually — e.g.
    postgres_port must be a valid int, caught here rather than at connection time.
    database_url is derived, not stored, so it can never drift out of sync
    with the individual fields.
    """

    # --- Postgres ---
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # --- Groq (kept for fallback / future use) ---
    groq_api_key: str = ""
    groq_model_name: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2048

    # --- Cerebras ---
    cerebras_api_key: str = ""
    cerebras_model_name: str = "llama-3.3-70b"
    cerebras_temperature: float = 0.1
    cerebras_max_tokens: int = 2048

    # --- Claude ---
    # Anthropic Messages API. $1/$5 per M tokens (Haiku 4.5).
    # 30-alert eval run ≈ $0.50 well within $5 budget.
    claude_api_key: str = ""
    claude_model_name: str = "claude-haiku-4-5-20251001"
    claude_temperature: float = 0.1
    claude_max_tokens: int = 2048

    # --- Gemini ---
    # OpenAI-compatible gateway (https://ai.google.dev/gemini-api/docs/openai).
    # Free tier: 15 RPM, 1,500 RPD, 1M TPM — no card, no expiration.
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 2048

    # --- AMLSim ---
    amlsim_output_dir: Path

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        """
        Assemble the SQLAlchemy connection string from validated parts.

        psycopg2 driver specified explicitly (postgresql+psycopg2://) rather
        than relying on SQLAlchemy's default, so the driver dependency is
        unambiguous and matches what's pinned in requirements.txt.
        """
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Singleton instance — import this everywhere instead of instantiating
# Settings() again, so .env is parsed once and every module shares the
# same validated config object.
settings = Settings()