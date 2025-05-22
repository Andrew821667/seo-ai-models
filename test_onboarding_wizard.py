
import sys
sys.path.append('.')

try:
    from seo_ai_models.models.freemium.onboarding.onboarding_wizard import OnboardingWizard
    
    ow = OnboardingWizard(user_id="test_user", plan="basic")
    print("✅ OnboardingWizard успешно импортирован и создан")
    
    methods_to_check = [
        '_load_user_preferences',
        '_create_sample_project',
        '_setup_notifications', 
        '_schedule_follow_up',
        '_track_onboarding_completion'
    ]
    
    for method in methods_to_check:
        if hasattr(ow, method):
            print(f"✅ Метод {method} присутствует")
        else:
            print(f"❌ Метод {method} отсутствует")
            
    print("\n🎉 OnboardingWizard готов!")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
