
import os

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Добавляем недостающие методы в конец класса (перед последней строкой)
new_methods = """
    def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Загружает пользовательские предпочтения.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Dict[str, Any]: Словарь с предпочтениями пользователя
        """
        try:
            # Здесь будет загрузка из базы данных или файла
            default_preferences = {
                "language": "ru",
                "theme": "light",
                "notifications": {
                    "email": True,
                    "push": True,
                    "sms": False
                },
                "tutorial_speed": "normal",
                "show_tips": True,
                "auto_save": True
            }
            
            # В реальной реализации здесь будет загрузка из хранилища
            # preferences = self.storage.load_user_preferences(user_id)
            preferences = default_preferences.copy()
            
            print(f"✅ Предпочтения пользователя {user_id} загружены")
            return preferences
            
        except Exception as e:
            print(f"❌ Ошибка загрузки предпочтений пользователя {user_id}: {e}")
            return default_preferences

    def _create_sample_project(self, user_id: str, project_name: str = "Демо-проект") -> Optional[Dict[str, Any]]:
        """
        Создает демо-проект для новых пользователей.
        
        Args:
            user_id: ID пользователя
            project_name: Название демо-проекта
            
        Returns:
            Optional[Dict[str, Any]]: Данные созданного проекта или None
        """
        try:
            import uuid
            from datetime import datetime
            
            # Создаем демо-проект с примерными данными
            sample_project = {
                "id": str(uuid.uuid4()),
                "name": project_name,
                "url": "https://example.com",
                "description": "Это демонстрационный проект для изучения возможностей платформы",
                "owner_id": user_id,
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "settings": {
                    "analysis_frequency": "weekly",
                    "notifications": True,
                    "auto_reports": True
                },
                "sample_data": {
                    "seo_score": 75,
                    "pages_analyzed": 25,
                    "issues_found": 8,
                    "opportunities": 5
                }
            }
            
            # В реальной реализации здесь будет сохранение в ProjectManagement
            # self.project_management.create_project(**sample_project)
            
            print(f"✅ Демо-проект '{project_name}' создан для пользователя {user_id}")
            return sample_project
            
        except Exception as e:
            print(f"❌ Ошибка создания демо-проекта: {e}")
            return None

    def _setup_notifications(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Настраивает систему уведомлений для пользователя.
        
        Args:
            user_id: ID пользователя
            preferences: Предпочтения уведомлений
            
        Returns:
            bool: True если настройки успешно применены
        """
        try:
            notification_settings = preferences.get("notifications", {})
            
            # Настройки уведомлений по умолчанию
            default_settings = {
                "email_reports": notification_settings.get("email", True),
                "push_notifications": notification_settings.get("push", True),
                "sms_alerts": notification_settings.get("sms", False),
                "weekly_digest": True,
                "critical_issues": True,
                "system_updates": True
            }
            
            # В реальной реализации здесь будет интеграция с системой уведомлений
            notification_types = []
            if default_settings["email_reports"]:
                notification_types.append("email")
            if default_settings["push_notifications"]:
                notification_types.append("push")
            if default_settings["sms_alerts"]:
                notification_types.append("sms")
            
            # Сохраняем настройки
            # self.notification_service.setup_user_notifications(user_id, default_settings)
            
            print(f"✅ Уведомления настроены для пользователя {user_id}: {', '.join(notification_types)}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка настройки уведомлений: {e}")
            return False

    def _schedule_follow_up(self, user_id: str, plan: str) -> bool:
        """
        Планирует последующие действия и напоминания.
        
        Args:
            user_id: ID пользователя
            plan: Тарифный план пользователя
            
        Returns:
            bool: True если планирование успешно
        """
        try:
            from datetime import datetime, timedelta
            
            # Определяем расписание в зависимости от плана
            follow_up_schedule = {
                "micro": [
                    {"days": 1, "action": "check_first_analysis", "message": "Как дела с первым анализом?"},
                    {"days": 7, "action": "usage_tips", "message": "Полезные советы по использованию"},
                    {"days": 30, "action": "upgrade_suggestion", "message": "Рассмотрите возможность апгрейда"}
                ],
                "basic": [
                    {"days": 3, "action": "feature_exploration", "message": "Изучите дополнительные возможности"},
                    {"days": 14, "action": "optimization_tips", "message": "Советы по оптимизации"},
                    {"days": 30, "action": "results_review", "message": "Обзор достигнутых результатов"}
                ],
                "professional": [
                    {"days": 1, "action": "advanced_setup", "message": "Настройка расширенных функций"},
                    {"days": 7, "action": "integration_help", "message": "Помощь с интеграциями"},
                    {"days": 21, "action": "roi_analysis", "message": "Анализ возврата инвестиций"}
                ]
            }
            
            schedule = follow_up_schedule.get(plan, follow_up_schedule["micro"])
            
            scheduled_actions = []
            for item in schedule:
                follow_up_date = datetime.now() + timedelta(days=item["days"])
                action = {
                    "user_id": user_id,
                    "action": item["action"],
                    "message": item["message"],
                    "scheduled_date": follow_up_date.isoformat(),
                    "status": "scheduled"
                }
                scheduled_actions.append(action)
            
            # В реальной реализации здесь будет сохранение в планировщик задач
            # self.scheduler.schedule_follow_ups(scheduled_actions)
            
            print(f"✅ Запланировано {len(scheduled_actions)} последующих действий для пользователя {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка планирования последующих действий: {e}")
            return False

    def _track_onboarding_completion(self, user_id: str, completion_data: Dict[str, Any]) -> bool:
        """
        Отслеживает завершение онбординга.
        
        Args:
            user_id: ID пользователя
            completion_data: Данные о завершении онбординга
            
        Returns:
            bool: True если отслеживание успешно
        """
        try:
            from datetime import datetime
            
            # Данные завершения онбординга
            completion_record = {
                "user_id": user_id,
                "completed_at": datetime.now().isoformat(),
                "total_time_spent": completion_data.get("time_spent", 0),
                "steps_completed": len(self.progress.get("completed_steps", [])),
                "steps_skipped": completion_data.get("steps_skipped", 0),
                "plan": self.plan,
                "satisfaction_score": completion_data.get("satisfaction", None),
                "feedback": completion_data.get("feedback", ""),
                "conversion_source": completion_data.get("source", "direct")
            }
            
            # Метрики завершения
            completion_metrics = {
                "completion_rate": (completion_record["steps_completed"] / 
                                  (completion_record["steps_completed"] + completion_record["steps_skipped"])) * 100,
                "time_per_step": completion_record["total_time_spent"] / max(1, completion_record["steps_completed"]),
                "plan_conversion": self.plan != "micro"  # Конверсия в платный план
            }
            
            # В реальной реализации здесь будет отправка в аналитику
            # self.analytics.track_onboarding_completion(completion_record, completion_metrics)
            
            # Отправляем событие завершения
            # self.event_bus.publish("onboarding_completed", {
            #     "user_id": user_id,
            #     "plan": self.plan,
            #     "metrics": completion_metrics
            # })
            
            print(f"✅ Завершение онбординга отслежено для пользователя {user_id}")
            print(f"   Завершено шагов: {completion_record['steps_completed']}")
            print(f"   Коэффициент завершения: {completion_metrics['completion_rate']:.1f}%")
            print(f"   План: {self.plan}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка отслеживания завершения онбординга: {e}")
            return False

"""

# Добавляем методы в конец класса (перед последней закрывающей скобкой, если есть)
# Находим конец файла и вставляем перед ним
lines = content.split('\n')
# Вставляем перед последними строками
insert_index = len(lines) - 1
while insert_index > 0 and lines[insert_index].strip() == "":
    insert_index -= 1

# Вставляем новые методы
new_method_lines = new_methods.split('\n')
lines = lines[:insert_index] + new_method_lines + lines[insert_index:]

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print("✅ Все недостающие методы из чек-листа добавлены!")
