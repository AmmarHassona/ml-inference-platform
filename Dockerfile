FROM python:3.10.19-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install uv
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]