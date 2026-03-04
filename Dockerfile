FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy dependency metadata first (cache-friendly layer)
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY packages/data/pyproject.toml packages/data/pyproject.toml
COPY packages/indicators/pyproject.toml packages/indicators/pyproject.toml
COPY packages/strategies/pyproject.toml packages/strategies/pyproject.toml
COPY packages/backtest/pyproject.toml packages/backtest/pyproject.toml
COPY packages/executor/pyproject.toml packages/executor/pyproject.toml
COPY examples/custom_strategy/pyproject.toml examples/custom_strategy/pyproject.toml

# Copy full source
COPY packages/ packages/
COPY examples/ examples/
COPY src/ src/
COPY scripts/ scripts/
COPY bot.py copybot.py copybot_v2.py backtest_engine.py ./

# Install all workspace packages (frozen = exact versions from lock file)
RUN uv sync --all-packages --frozen --no-dev

# Default: paper trading bot
ENTRYPOINT ["uv", "run", "python", "-u"]
CMD ["bot.py", "--paper"]
