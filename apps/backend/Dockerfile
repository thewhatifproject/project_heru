FROM python:3.11-slim

WORKDIR /workspace
COPY apps/backend/pyproject.toml apps/backend/README.md /workspace/apps/backend/
COPY apps/backend/app /workspace/apps/backend/app
COPY core/streamdiffusionv2 /workspace/core/streamdiffusionv2

WORKDIR /workspace/apps/backend
RUN pip install --no-cache-dir -e .

EXPOSE 8000
ENV STREAMDIFFUSIONV2_PATH=/workspace/core/streamdiffusionv2
CMD ["python", "-m", "app.main"]
