
import os

# Читаем файл
with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем заглушку в update_project (строка 315)
# Найдем метод update_project и заменим заглушку
lines = content.split('\n')

# Найдем строку с return None и заменим её на полноценную реализацию
for i, line in enumerate(lines):
    if i >= 314 and i <= 316 and "return None" in line and "project = self.get_project(project_id)" in lines[i-3]:
        # Это заглушка в update_project, заменяем её
        lines[i] = """        
        # Сохраняем обновленный проект
        self._save_project(project)
        
        print(f"✅ Проект {project_id} успешно обновлен")
        return project"""
        break

# Записываем обратно
with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка #1 (update_project) исправлена!")
