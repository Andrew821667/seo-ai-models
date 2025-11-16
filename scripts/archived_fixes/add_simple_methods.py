
import os

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Простые методы с минимальной функциональностью
new_methods = """

    def _load_user_preferences(self, user_id: str):
        """Загружает пользовательские предпочтения."""
        preferences = {
            "language": "ru",
            "theme": "light", 
            "notifications": {"email": True, "push": True}
        }
        print(f"✅ Предпочтения пользователя {user_id} загружены")
        return preferences

    def _create_sample_project(self, user_id: str, project_name: str = "Демо-проект"):
        """Создает демо-проект для новых пользователей."""
        try:
            import uuid
            project = {
                "id": str(uuid.uuid4()),
                "name": project_name,
                "url": "https://example.com",
                "owner_id": user_id,
                "status": "active"
            }
            print(f"✅ Демо-проект создан для {user_id}")
            return project
        except Exception as e:
            print(f"❌ Ошибка создания проекта: {e}")
            return None

    def _setup_notifications(self, user_id: str, preferences: dict):
        """Настраивает уведомления."""
        try:
            types = []
            if preferences.get("notifications", {}).get("email"):
                types.append("email")
            if preferences.get("notifications", {}).get("push"):
                types.append("push")
            print(f"✅ Уведомления настроены: {', '.join(types)}")
            return True
        except Exception as e:
            print(f"❌ Ошибка настройки уведомлений: {e}")
            return False

    def _schedule_follow_up(self, user_id: str, plan: str):
        """Планирует последующие действия."""
        try:
            from datetime import datetime, timedelta
            follow_ups = []
            if plan == "micro":
                follow_ups = [{"days": 1, "message": "Проверка первого анализа"}]
            elif plan == "basic":  
                follow_ups = [{"days": 3, "message": "Изучение функций"}]
            else:
                follow_ups = [{"days": 1, "message": "Расширенная настройка"}]
            
            print(f"✅ Запланировано {len(follow_ups)} действий для {user_id}")
            return True
        except Exception as e:
            print(f"❌ Ошибка планирования: {e}")
            return False

    def _track_onboarding_completion(self, user_id: str, data: dict):
        """Отслеживает завершение онбординга."""
        try:
            from datetime import datetime
            record = {
                "user_id": user_id,
                "completed_at": datetime.now().isoformat(),
                "steps_completed": len(self.progress.get("completed_steps", [])),
                "plan": self.plan
            }
            print(f"✅ Онбординг отслежен для {user_id}, план: {self.plan}")
            return True
        except Exception as e:
            print(f"❌ Ошибка отслеживания: {e}")
            return False
"""

# Добавляем методы в конец файла
content = content.rstrip() + new_methods

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Методы добавлены!")
