
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем заглушку в update_template (предположительно)
# Найдем и заменим вторую заглушку return None
lines = content.split('\n')

for i, line in enumerate(lines):
    if i >= 700 and i <= 702 and "return None" in line.strip() and line.strip() == "return None":
        # Заменяем заглушку
        lines[i] = """        # Сохраняем обновленный шаблон
        self._save_template(template)
        
        print(f"✅ Шаблон {template_id} успешно обновлен")
        return template"""
        break

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка #2 исправлена!")
