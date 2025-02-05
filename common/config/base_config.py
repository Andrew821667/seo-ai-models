from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

class BaseConfig(BaseModel):
    """Базовый класс для всех конфигураций"""
    
    def save(self, path: str) -> None:
        """
        Сохранение конфигурации в JSON файл
        Args:
            path: путь для сохранения
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(
                self.model_dump(),
                f,
                indent=4,
                ensure_ascii=False
            )
    
    @classmethod
    def load(cls, path: str) -> 'BaseConfig':
        """
        Загрузка конфигурации из JSON файла
        Args:
            path: путь к файлу
        Returns:
            объект конфигурации
        """
        with open(path, encoding='utf-8') as f:
            return cls(**json.load(f))
    
    class Config:
        """Конфигурация Pydantic модели"""
        arbitrary_types_allowed = True
