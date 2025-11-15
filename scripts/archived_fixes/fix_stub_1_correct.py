
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Заменяем строку 315 (индекс 314) - это заглушка в update_project
if len(lines) > 314 and "return None" in lines[314] and lines[314].strip() == "return None":
    # Находим конец метода update_project и вставляем правильную логику
    for i in range(315, len(lines)):
        if lines[i].strip().startswith("def ") and not lines[i].strip().startswith("def update_project"):
            # Это начало следующего метода, вставляем код перед ним
            lines[314] = "        # Сохраняем обновленный проект\n"
            lines.insert(315, "        self._save_project(project)\n")
            lines.insert(316, "        \n")
            lines.insert(317, "        print(f\"✅ Проект {project_id} успешно обновлен\")\n")
            lines.insert(318, "        return project\n")
            lines.insert(319, "\n")
            break

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ Заглушка #1 (update_project) исправлена правильно!")
