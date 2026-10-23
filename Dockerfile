FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/matplotlib INDIA_WEATHER_ROOT=/app
WORKDIR /app
COPY requirements.txt requirements-app.txt requirements-app.lock.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements-app.lock.txt
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && useradd --create-home weather
COPY app.py frontend.py ./
COPY models/correction ./models/correction
RUN mkdir -p /app/data/live_cache && chown -R weather:weather /app/data
USER weather
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3)"
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1"]
