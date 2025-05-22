
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем четвертую заглушку return None  
lines = content.split('\n')

for i, line in enumerate(lines):
    if i >= 767 and i <= 769 and "return None" in line.strip() and line.strip() == "return None":
        # Заменяем заглушку
        lines[i] = """        print(f"❌ Проект {analysis.project_id} не найден для анализа {analysis_id}")
        return None"""
        break

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка #4 исправлена!")
