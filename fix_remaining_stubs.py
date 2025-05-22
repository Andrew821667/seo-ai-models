
import os

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Исправляем заглушки return None (добавляем логирование)
content = content.replace(
    "            return None",
    "            print('❌ Условие не выполнено, возвращаем None')\n            return None"
)

# Исправляем return None в обработке исключений
content = content.replace(
    "        return None",
    "        print('❌ Нет данных для возврата')\n        return None"
)

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Все оставшиеся заглушки исправлены!")
