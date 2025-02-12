from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import numpy as np
import logging
from pathlib import Path

from common.config.dim_reducer_config import DimReducerConfig
from .model import DimensionReducer
from .inference import DimReducerInference
from common.utils.monitoring import track_inference_time
from common.utils.cache import Cache

logger = logging.getLogger(__name__)

# Модели данных
class TextInput(BaseModel):
    """Входные данные для анализа текста"""
    text: str = Field(..., min_length=1, description="Текст для анализа")
    html: Optional[str] = Field(None, description="HTML разметка текста")
    return_importance: bool = Field(True, description="Возвращать ли важность признаков")

class BatchTextInput(BaseModel):
    """Входные данные для пакетной обработки"""
    texts: List[str] = Field(..., min_items=1, description="Список текстов для анализа")
    html_texts: Optional[List[str]] = Field(None, description="Список HTML разметок")

class DimensionReducerResult(BaseModel):
    """Результаты сжатия размерности"""
    latent_features: List[float] = Field(..., description="Сжатые признаки")
    feature_importance: Optional[List[float]] = Field(None, description="Важность признаков")
    reconstruction_error: float = Field(..., description="Ошибка реконструкции")

# Создание роутера
router = APIRouter(prefix="/api/v1/dim-reducer", tags=["dimension-reducer"])

# Глобальные объекты
MODEL = None
CONFIG = None
CACHE = Cache()

def get_model():
    """Получение инициализированной модели"""
    global MODEL, CONFIG
    if MODEL is None:
        try:
            CONFIG = DimReducerConfig()
            model_path = Path("models/dim_reducer/checkpoints/final_model.pt")
            if not model_path.exists():
                raise FileNotFoundError("Model file not found")
            
            MODEL = DimReducerInference(
                str(model_path),
                CONFIG,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise HTTPException(
                status_code=500,
                detail="Model initialization failed"
            )
    return MODEL

@router.post("/analyze", response_model=DimensionReducerResult)
@track_inference_time
async def analyze_text(
    input_data: TextInput,
    model: DimReducerInference = Depends(get_model)
):
    """
    Анализ текста с помощью DimensionReducer
    
    - Извлекает SEO-характеристики
    - Выполняет сжатие размерности
    - Оценивает важность признаков
    """
    try:
        # Проверяем кэш
        cache_key = f"dim_reducer:{hash(input_data.text)}"
        if cached_result := CACHE.get(cache_key):
            return cached_result
        
        # Обработка текста
        results = model.process_text(
            text=input_data.text,
            html=input_data.html
        )
        
        # Формируем ответ
        response = {
            'latent_features': results['latent_features'].tolist(),
            'reconstruction_error': float(results.get('reconstruction_error', 0.0))
        }
        
        if input_data.return_importance:
            response['feature_importance'] = results['feature_importance'].tolist()
            
        # Кэшируем результат
        CACHE.set(cache_key, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/batch-analyze", response_model=List[DimensionReducerResult])
@track_inference_time
async def batch_analyze(
    input_data: BatchTextInput,
    model: DimReducerInference = Depends(get_model)
):
    """
    Пакетный анализ текстов
    """
    try:
        results = []
        for idx, text in enumerate(input_data.texts):
            # Проверяем кэш для каждого текста
            cache_key = f"dim_reducer:{hash(text)}"
            if cached_result := CACHE.get(cache_key):
                results.append(cached_result)
                continue
                
            # Обработка текста
            html = input_data.html_texts[idx] if input_data.html_texts else None
            result = model.process_text(text=text, html=html)
            
            # Формируем ответ
            response = {
                'latent_features': result['latent_features'].tolist(),
                'feature_importance': result['feature_importance'].tolist(),
                'reconstruction_error': float(result.get('reconstruction_error', 0.0))
            }
            
            # Кэшируем результат
            CACHE.set(cache_key, response)
            results.append(response)
            
        return results
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/model-info")
async def get_model_info(
    model: DimReducerInference = Depends(get_model)
):
    """
    Получение информации о модели
    """
    try:
        return {
            'config': model.config.dict(),
            'device': str(next(model.model.parameters()).device),
            'input_dim': model.config.input_dim,
            'latent_dim': model.config.latent_dim
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
