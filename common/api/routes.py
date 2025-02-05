from fastapi import FastAPI, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
import torch
import logging
import json
from datetime import datetime

# Импорты для SEO Advisor
from ..core.models.advisor import SEOAdvisor
from ..config.advisor_config import AdvisorConfig
from ..monitoring.performance import PerformanceMonitor

# Импорты для DimensionReducer
from ..core.models.dim_reducer.model import DimensionReducer
from ..core.models.dim_reducer.inference import DimReducerInference
from ..config.dim_reducer_config import DimReducerConfig

# Инициализация логгера
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="SEO Models API",
    description="API для SEO рекомендаций и оптимизации",
    version="1.0.0"
)

# Модели данных для SEO Advisor
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

# Модели данных для DimensionReducer
class DimReducerInput(BaseModel):
    """Входные данные для анализа текста"""
    text: str
    html: Optional[str] = None
    return_importance: bool = True

class DimReducerBatchInput(BaseModel):
    """Входные данные для пакетной обработки"""
    texts: List[str]
    html_texts: Optional[List[str]] = None

class DimReducerResult(BaseModel):
    """Результаты сжатия размерности"""
    latent_features: List[float]
    feature_importance: Optional[List[float]]
    reconstruction_error: float

# Глобальные объекты
seo_advisor_model = None
dim_reducer_model = None
advisor_config = None
dim_reducer_config = None
monitor = PerformanceMonitor()

def get_advisor_model():
    """Получение инициализированной модели SEO Advisor"""
    global seo_advisor_model, advisor_config
    if seo_advisor_model is None:
        try:
            advisor_config = AdvisorConfig()
            seo_advisor_model = SEOAdvisor(advisor_config)
            seo_advisor_model.load_state_dict(
                torch.load('models/seo_advisor.pth')
            )
            seo_advisor_model.eval()
        except Exception as e:
            logger.error(f"Ошибка при загрузке SEO Advisor: {e}")
            raise HTTPException(
                status_code=500,
                detail="Ошибка инициализации SEO Advisor"
            )
    return seo_advisor_model

def get_dim_reducer_model():
    """Получение инициализированной модели DimensionReducer"""
    global dim_reducer_model, dim_reducer_config
    if dim_reducer_model is None:
        try:
            dim_reducer_config = DimReducerConfig()
            dim_reducer_model = DimReducerInference(
                'models/dim_reducer/final_model.pt',
                dim_reducer_config,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        except Exception as e:
            logger.error(f"Ошибка при загрузке DimensionReducer: {e}")
            raise HTTPException(
                status_code=500,
                detail="Ошибка инициализации DimensionReducer"
            )
    return dim_reducer_model

# Маршруты SEO Advisor
@app.post("/advisor/analyze", response_model=AnalysisResponse, tags=["seo-advisor"])
async def analyze_content(
    content: ContentInput,
    model: SEOAdvisor = Depends(get_advisor_model)
):
    """Анализ контента и генерация рекомендаций"""
    try:
        start_time = datetime.now()
        
        inputs = {
            'text': content.text,
            'language': content.language,
            'keywords': content.keywords
        }
        
        with torch.no_grad():
            outputs = model(inputs)
            
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

# Маршруты DimensionReducer
@app.post("/dim-reducer/analyze", response_model=DimReducerResult, tags=["dimension-reducer"])
async def reduce_dimensions(
    input_data: DimReducerInput,
    model: DimReducerInference = Depends(get_dim_reducer_model)
):
    """Сжатие размерности и анализ текста"""
    try:
        start_time = datetime.now()
        
        results = model.process_text(
            text=input_data.text,
            html=input_data.html
        )
        
        response = {
            'latent_features': results['latent_features'].tolist(),
            'feature_importance': results['feature_importance'].tolist() if input_data.return_importance else None,
            'reconstruction_error': float(results.get('reconstruction_error', 0.0))
        }
        
        end_time = datetime.now()
        monitor.track_inference(start_time, end_time)
        monitor.track_memory()
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка при обработке текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dim-reducer/batch", response_model=List[DimReducerResult], tags=["dimension-reducer"])
async def batch_reduce_dimensions(
    input_data: DimReducerBatchInput,
    model: DimReducerInference = Depends(get_dim_reducer_model)
):
    """Пакетная обработка текстов"""
    try:
        start_time = datetime.now()
        
        results = []
        for idx, text in enumerate(input_data.texts):
            html = input_data.html_texts[idx] if input_data.html_texts else None
            result = model.process_text(text=text, html=html)
            
            results.append({
                'latent_features': result['latent_features'].tolist(),
                'feature_importance': result['feature_importance'].tolist(),
                'reconstruction_error': float(result.get('reconstruction_error', 0.0))
            })
        
        end_time = datetime.now()
        monitor.track_inference(start_time, end_time)
        monitor.track_memory()
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка при пакетной обработке: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Общие маршруты
@app.get("/metrics")
async def get_metrics():
    """Получение метрик производительности"""
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
    """Проверка работоспособности сервиса"""
    return {
        "status": "healthy",
        "seo_advisor_loaded": seo_advisor_model is not None,
        "dim_reducer_loaded": dim_reducer_model is not None,
        "timestamp": datetime.now().isoformat()
    }
