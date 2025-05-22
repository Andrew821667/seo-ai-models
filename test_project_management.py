
import sys
sys.path.append('.')

try:
    from seo_ai_models.web.dashboard.project_management import ProjectManagement
    
    # Создаем экземпляр
    pm = ProjectManagement(data_dir="test_data")
    print("✅ ProjectManagement успешно импортирован и создан")
    
    # Проверяем наличие всех методов из чек-листа
    methods_to_check = [
        'update_project_status',
        'schedule_analysis', 
        'get_project_analyses',
        'update_analysis_status'
    ]
    
    for method in methods_to_check:
        if hasattr(pm, method):
            print(f"✅ Метод {method} присутствует")
        else:
            print(f"❌ Метод {method} отсутствует")
            
    print("\n🎉 Все проверки завершены!")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
