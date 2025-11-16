
import os

with open("seo_ai_models/web/dashboard/project_management.py", "r", encoding="utf-8") as f:
    content = f.read()

# Найдем место для вставки update_project_status (правильно отформатированного)
new_method = """
    def update_project_status(self, project_id: str, status: str) -> Optional[Project]:
        """
        Обновляет статус проекта.
        
        Args:
            project_id: ID проекта  
            status: Новый статус (active, archived, deleted, etc.)
            
        Returns:
            Optional[Project]: Обновленный проект, если найден, иначе None
        """
        project = self.get_project(project_id)
        if not project:
            print(f'❌ Проект {project_id} не найден')
            return None
            
        old_status = project.status
        project.status = status
        project.updated_at = datetime.now()
        
        # Сохраняем изменения
        self._save_project(project)
        
        print(f'✅ Статус проекта {project_id} изменен с \'{old_status}\' на \'{status}\'')
        return project

"""

# Заменяем обычной заменой строки
content = content.replace(
    "    def delete_project(self, project_id: str) -> bool:",
    new_method + "    def delete_project(self, project_id: str) -> bool:"
)

with open("seo_ai_models/web/dashboard/project_management.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Метод update_project_status добавлен с правильными отступами!")
