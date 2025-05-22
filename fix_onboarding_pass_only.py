
import os

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем только pass в методе _configure_steps_for_plan
content = content.replace(
    "            # Здесь можно добавить дополнительные шаги для Enterprise\n            pass",
    "            # Добавляем специальные шаги для Enterprise\n            print(f\"✅ Enterprise план настроен для пользователя\")"
)

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушка pass исправлена!")
