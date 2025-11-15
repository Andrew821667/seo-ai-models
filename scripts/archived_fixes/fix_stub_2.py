
import os

# Читаем файл
with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split('\n')

# Найдем заглушку в create_analysis (около строки 372)
for i, line in enumerate(lines):
    if i >= 370 and i <= 375 and "return None" in line and "project = self.get_project(project_id)" in lines[i-3]:
        # Заменяем заглушку на полную реализацию
        lines[i] = """        
        # Создаем анализ
        analysis = Analysis(
            analysis_id=analysis_id,
            project_id=project_id,
            name=name,
            type=analysis_type,
            settings=settings or {},
            status="pending",
            created_at=datetime.now()
        )
        
        # Сохраняем анализ
        self._save_analysis(analysis)
        
        print(f"✅ Анализ {analysis_id} создан для проекта {project_id}")
        return analysis"""
        break

# Записываем обратно
with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка #2 (create_analysis) исправлена!")
