
import sys
sys.path.append('.')

try:
    from seo_ai_models.web.dashboard.report_generator import ReportGenerator
    
    rg = ReportGenerator(data_dir="test_data")
    print("✅ ReportGenerator успешно импортирован и создан")
    
    methods_to_check = [
        '_generate_executive_summary',
        '_generate_recommendations_section', 
        '_apply_custom_styling',
        '_add_interactive_elements'
    ]
    
    for method in methods_to_check:
        if hasattr(rg, method):
            print(f"✅ Метод {method} присутствует")
        else:
            print(f"❌ Метод {method} отсутствует")
            
    print("\n🎉 ReportGenerator готов!")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
