FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && \
  FFMPEG_VERSION="$(apt-cache policy ffmpeg | grep Candidate | awk '{print $2}' | head -1)" && \
  apt-get install -y --no-install-recommends \
  "ffmpeg=${FFMPEG_VERSION}" \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked

COPY assets /app/assets
COPY src /app/src

CMD ["uv", "run", "python", "-m", "src.main"]
