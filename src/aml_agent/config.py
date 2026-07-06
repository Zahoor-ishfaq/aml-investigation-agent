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
    
    # --- Groq ---
    groq_api_key: str
    groq_model_name: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2048
    # --- AMLSim ---
    # Path type gives startup validation: pydantic converts the string to a Path
    # object and downstream code can .exists() / .iterdir() without re-parsing.
    # We deliberately do NOT verify the directory exists here — that would
    # couple config load to filesystem state and fail in test environments
    # where AMLSim isn't installed. Ingestion script checks existence.
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