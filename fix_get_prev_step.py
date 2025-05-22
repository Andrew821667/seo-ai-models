
import os

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем return None в _get_prev_step и в обработке ошибок
content = content.replace(
    """        except (ValueError, IndexError):
            # Если что-то пошло не так, возвращаем None
            return None""",
    """        except (ValueError, IndexError) as e:
            # Если что-то пошло не так, логируем и возвращаем None
            print(f"❌ Ошибка определения следующего шага: {e}")
            return None"""
)

content = content.replace(
    """        if not completed_steps:
            return None""",
    """        if not completed_steps:
            print("❌ Нет завершенных шагов, предыдущий шаг недоступен")
            return None"""
)

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушки в _get_prev_step исправлены!")
