FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data/demo_fixture_pack ./data/demo_fixture_pack

# The judge endpoint intentionally uses frozen retrieval inputs so OpenCV 5,
# policy behavior, failure handling, and human control can be reproduced with
# a small deterministic image. Full OpenCLIP evaluation remains an offline job.
RUN pip install --no-cache-dir \
      "boto3>=1.35,<2" \
      "fastapi>=0.115,<1" \
      "numpy>=1.26,<3" \
      "opencv-python-headless>=5,<6" \
      "pandas>=2.1,<4" \
      "Pillow>=10,<13" \
      "PyYAML>=6,<7" \
      "uvicorn>=0.30,<1" \
    && pip install --no-cache-dir --no-deps .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).read()"

CMD ["uvicorn", "bridgetrend_vision.competition_app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
