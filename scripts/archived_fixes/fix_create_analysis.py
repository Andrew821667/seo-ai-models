
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменим заглушку в create_analysis
content = content.replace(
    """        # Сохраняем анализ
        self.analyses[analysis_id] = analysis
        project.analyses.append(analysis_id)
        self._save_analysis(analysis)
        return None""",
    """        # Сохраняем анализ
        self.analyses[analysis_id] = analysis
        project.analyses.append(analysis_id)
        self._save_analysis(analysis)
        
        print(f"✅ Анализ '{name}' создан для проекта {project_id}")
        return analysis"""
)

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушка в create_analysis исправлена!")
