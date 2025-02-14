
import sys
import importlib

# Попытка принудительной перезагрузки модуля
sys.path.append('/content/seo-ai-models')

# Явная перезагрузка модулей
import common.config.advisor_config
importlib.reload(common.config.advisor_config)

try:
    from common.config.advisor_config import AdvisorConfig, ModelConfig
    
    print("✅ Импорт успешен!")
    
    # Создаем тестовые конфигурации
    advisor_config = AdvisorConfig(
        model_path='/tmp/model'
    )
    model_config = ModelConfig()

    print("\nТестовые конфигурации:")
    print("AdvisorConfig:", advisor_config)
    print("ModelConfig:", model_config)

except Exception as e:
    print(f"❌ Ошибка импорта: {e}")
    
    # Расширенная диагностика
    import traceback
    traceback.print_exc()
    
    # Проверка содержимого модуля
    import common.config.advisor_config
    print("\nСодержимое модуля:")
    print(dir(common.config.advisor_config))
