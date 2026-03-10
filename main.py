"""
Entrypoint for the lgrap LangGraph agent API.

Run directly:
    python main.py

Or via the installed console script:
    lgrap

For development with hot-reload:
    uvicorn main:app --reload
"""
import logging

import uvicorn

from app.api import create_app
from app.utils.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# Module-level `app` lets `uvicorn main:app --reload` work without changes.
app = create_app()


def main() -> None:
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
