
import os

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем первую return None в _get_next_step
content = content.replace(
    """        if not next_steps:
            return None""",
    """        if not next_steps:
            print("❌ Нет доступных следующих шагов")
            return None"""
)

# Заменяем вторую return None в _get_next_step
content = content.replace(
    """        if current_step == OnboardingStep.COMPLETE.value:
            return None""",
    """        if current_step == OnboardingStep.COMPLETE.value:
            print("✅ Онбординг завершен, следующих шагов нет")
            return None"""
)

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушки в _get_next_step исправлены!")
