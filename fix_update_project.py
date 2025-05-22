
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

# Найдем точную позицию заглушки и заменим её
content = content.replace(
    """        project.updated_at = datetime.now()
        
        # Сохраняем проект
        self._save_project(project)
        return None""",
    """        project.updated_at = datetime.now()
        
        # Сохраняем проект
        self._save_project(project)
        
        print(f"✅ Проект {project_id} успешно обновлен")
        return project"""
)

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушка в update_project исправлена!")
