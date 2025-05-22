
import os

# Читаем основной файл
with open("seo_ai_models/web/dashboard/project_management.py", "r") as f:
    content = f.read()

# Читаем методы
with open("update_project_status_method.txt", "r") as f:
    update_method = f.read()

with open("schedule_analysis_method.txt", "r") as f:
    schedule_method = f.read()

# Найдем место для вставки - перед методом delete_project
lines = content.split('\n')
new_lines = []

for i, line in enumerate(lines):
    if line.strip().startswith("def delete_project("):
        # Вставляем новые методы перед delete_project
        new_lines.extend(update_method.split('\n'))
        new_lines.append('')
        new_lines.extend(schedule_method.split('\n'))  
        new_lines.append('')
    new_lines.append(line)

# Записываем обратно
with open("seo_ai_models/web/dashboard/project_management.py", "w") as f:
    f.write('\n'.join(new_lines))

print("✅ Методы добавлены в файл!")
