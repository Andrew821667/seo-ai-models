
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Найдем заглушку в create_analysis (должна быть около строки 372)
for i, line in enumerate(lines):
    if i >= 370 and i <= 375 and "return None" in line.strip() and line.strip() == "return None":
        # Проверим, что мы в методе create_analysis
        method_found = False
        for j in range(max(0, i-15), i):
            if "def create_analysis" in lines[j]:
                method_found = True
                break
        
        if method_found:
            # Заменяем заглушку
            lines[i] = """        # Создаем анализ
        analysis = Analysis(
            analysis_id=analysis_id,
            project_id=project_id,
            name=name,
            type=analysis_type,
            settings=settings or {},
            status="pending",
            created_at=datetime.now()
        )
        
        # Добавляем анализ в коллекцию
        self.analyses[analysis_id] = analysis
        
        # Связываем с проектом (если у проекта есть список анализов)
        if hasattr(project, 'analyses') and hasattr(project.analyses, 'append'):
            project.analyses.append(analysis_id)
        
        # Сохраняем анализ
        self._save_analysis(analysis)
        
        print(f"✅ Анализ '{name}' создан для проекта {project_id}")
        return analysis
"""
            break

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ Заглушка #2 (create_analysis) исправлена!")
