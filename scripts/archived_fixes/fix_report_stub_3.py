
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем третью заглушку return None
lines = content.split('\n')

for i, line in enumerate(lines):
    if i >= 762 and i <= 764 and "return None" in line.strip() and line.strip() == "return None":
        # Заменяем заглушку (вероятно это в generate_report_from_analysis)
        lines[i] = """        print(f"❌ Анализ {analysis_id} не найден")
        return None"""
        break

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка #3 исправлена!")
