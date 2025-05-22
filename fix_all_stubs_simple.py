
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

# Исправляем первую заглушку (update_project)
content = content.replace(
    "        return None",
    "        return project",
    1  # заменяем только первое вхождение
)

# Исправляем вторую заглушку (create_analysis)  
content = content.replace(
    "        return None",
    "        return analysis",
    1  # заменяем следующее вхождение
)

# Исправляем третью заглушку (update_analysis_status)
content = content.replace(
    "        return None", 
    "        return analysis",
    1  # заменяем последнее вхождение
)

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Все заглушки return None исправлены!")
