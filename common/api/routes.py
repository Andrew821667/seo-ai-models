from fastapi import FastAPI, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
import torch
import logging
import json
from datetime import datetime

from ..core.models.advisor import SEOAdvisor
from ..config.advisor_config import AdvisorConfig
from ..monitoring.performance import PerformanceMonitor

# Инициализация логгера
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="SEO Advisor API",
    description="API для SEO рекомендаций и оптимизации",
    version="1.0.0"
)

# Модели данных
class ContentInput(BaseModel):
    """Входные данные для анализа"""
    text: str
    url: Optional[str] = None
    language: Optional[str] = None
    keywords: Optional[List[str]] = None

class AnalysisResponse(BaseModel):
    """Ответ с результатами анализа"""
    rank_score: float
    suggestions: List[str]
    importance_scores: List[float]
    metrics: dict

# Глобальные объекты
model = None
config = None
monitor = PerformanceMonitor()

def get_model():
    """Получение инициализированной модели"""
    global model, config
    if model is None:
        try:
            config = AdvisorConfig()
            model = SEOAdvisor(config)
            model.load_state_dict(
                torch.load('models/seo_advisor.pth')
            )
            model.eval()
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise HTTPException(
                status_code=500,
                detail="Ошибка инициализации модели"
            )
    return model

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    content: ContentInput,
    model: SEOAdvisor = Depends(get_model)
):
    """
    Анализ контента и генерация рекомендаций
    """
    try:
        start_time = datetime.now()
        
        # Подготовка входных данных
        inputs = {
            'text': content.text,
            'language': content.language,
            'keywords': content.keywords
        }
        
        # Получение предсказаний
        with torch.no_grad():
            outputs = model(inputs)
            
        # Формирование ответа
        response = AnalysisResponse(
            rank_score=float(outputs['rank_score'].mean()),
            suggestions=model.decode_suggestions(
                outputs['suggestions']
            ),
            importance_scores=[
                float(score) for score in outputs['importance_scores']
            ],
            metrics=json.loads(
                json.dumps(outputs['text_metrics'])
            )
        )
        
        # Мониторинг производительности
        end_time = datetime.now()
        monitor.track_inference(start_time, end_time)
        monitor.track_memory()
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка при анализе контента: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/metrics")
async def get_metrics():
    """
    Получение метрик производительности
    """
    try:
        return monitor.get_performance_stats()
    except Exception as e:
        logger.error(f"Ошибка при получении метрик: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """
    Проверка работоспособности сервиса
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
