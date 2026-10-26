"""
Write the OpenAPI schema to `openapi.json` at the repo root, without running
the server. This file is the API/client contract: WeLearn-client generates its
types from it. Run with `make openapi-poetry`.
"""

import configparser
import os
from pathlib import Path


def _load_test_env() -> None:
    # the schema does not depend on secrets: reuse the dummy values from
    # pytest.ini so the export works without a .env
    config = configparser.ConfigParser()
    config.read(Path(__file__).resolve().parents[1] / "pytest.ini")
    for line in config["pytest"]["env"].strip().splitlines():
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    _load_test_env()

    from src.app.shared.api.openapi import OPENAPI_SCHEMA_PATH, render_openapi_schema
    from src.main import app

    OPENAPI_SCHEMA_PATH.write_text(render_openapi_schema(app), encoding="utf-8")
    print(f"OpenAPI schema written to {OPENAPI_SCHEMA_PATH}")


if __name__ == "__main__":
    main()
