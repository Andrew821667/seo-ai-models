
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split('\n')

# Находим метод update_project и заменяем заглушку
for i, line in enumerate(lines):
    # Ищем точную заглушку в update_project  
    if (i >= 314 and i <= 316 and 
        "return None" in line and 
        i > 300 and 
        any("def update_project" in lines[j] for j in range(max(0, i-30), i))):
        
        lines[i] = """        # Сохраняем обновленный проект
        self._save_project(project)
        
        print(f"✅ Проект {project_id} успешно обновлен")
        return project"""
        break

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка в update_project исправлена!")
