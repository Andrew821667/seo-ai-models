import logging.config
import json
from pathlib import Path
from typing import Optional

def setup_logging(
    default_level: int = logging.INFO,
    log_file: Optional[str] = None,
    config_file: Optional[str] = None
):
    """
    Настройка логирования
    Args:
        default_level: уровень логирования по умолчанию
        log_file: путь к файлу логов
        config_file: путь к файлу конфигурации логирования
    """
    if config_file and Path(config_file).exists():
        with open(config_file, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                *(
                    [logging.FileHandler(log_file)]
                    if log_file
                    else []
                )
            ]
        )

def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера с заданным именем
    Args:
        name: имя логгера
    Returns:
        настроенный логгер
    """
    return logging.getLogger(name)
