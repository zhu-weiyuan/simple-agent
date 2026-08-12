"""Root conftest: set env vars before any test module imports app_prod."""
import os

# CI has no LLM server; health endpoint should still report ok=True
# when LLM is not required.
os.environ.setdefault("REQUIRE_LLM_FOR_READINESS", "false")
