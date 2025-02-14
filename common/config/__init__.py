
from .advisor_config import AdvisorConfig, ModelConfig

__all__ = ['AdvisorConfig', 'ModelConfig']

# Явное добавление классов в текущее пространство имен
globals()['AdvisorConfig'] = AdvisorConfig
globals()['ModelConfig'] = ModelConfig
