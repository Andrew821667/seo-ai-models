# Multi-stage build для оптимизации размера образа
FROM python:3.10-slim as builder

# Устанавливаем системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libxml2-dev \
    libxslt1-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копируем файлы зависимостей
COPY requirements.txt web_requirements.txt ./

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r web_requirements.txt

# Production stage
FROM python:3.10-slim

# Устанавливаем runtime зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем виртуальное окружение из builder
COPY --from=builder /opt/venv /opt/venv

# Устанавливаем Playwright browsers
RUN /opt/venv/bin/pip install playwright && \
    /opt/venv/bin/playwright install chromium && \
    /opt/venv/bin/playwright install-deps chromium

# Создаем пользователя для запуска приложения
RUN useradd -m -u 1000 seoai && \
    mkdir -p /app && \
    chown -R seoai:seoai /app

WORKDIR /app

# Копируем код приложения
COPY --chown=seoai:seoai . .

# Устанавливаем пакет
RUN /opt/venv/bin/pip install -e .

# Переключаемся на непривилегированного пользователя
USER seoai

# Используем виртуальное окружение
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Порт для API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Команда по умолчанию
CMD ["uvicorn", "seo_ai_models.web.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
