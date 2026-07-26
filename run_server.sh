#!/bin/bash
# Helper script to launch FastAPI server using the project virtual environment
source .venv/bin/activate
uvicorn app:app --reload --port 8000
