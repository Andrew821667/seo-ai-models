
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем точную строку в update_template  
content = content.replace(
    """        template.updated_at = datetime.now()
        
        # Сохраняем шаблон
        self._save_template(template)
        return None""",
    """        template.updated_at = datetime.now()
        
        # Сохраняем шаблон
        self._save_template(template)
        
        print(f"✅ Шаблон {template_id} успешно обновлен")
        return template"""
)

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушка #2 (update_template) исправлена правильно!")
