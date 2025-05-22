
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split('\n')

# Находим заглушку в create_analysis
for i, line in enumerate(lines):
    if (i >= 370 and i <= 375 and 
        "return None" in line and
        any("def create_analysis" in lines[j] for j in range(max(0, i-20), i))):
        
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
        
        # Связываем с проектом
        if hasattr(project, 'analyses'):
            project.analyses.append(analysis_id)
        
        # Сохраняем анализ
        self._save_analysis(analysis)
        
        print(f"✅ Анализ '{name}' создан для проекта {project_id}")
        return analysis"""
        break

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Заглушка в create_analysis исправлена!")
