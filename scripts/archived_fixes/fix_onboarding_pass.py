
import os

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Заменяем pass на реальную логику для Enterprise плана
content = content.replace(
    """        # Для Enterprise плана добавляем дополнительные шаги, если они будут в будущем
        if self.plan == "enterprise":
            # Здесь можно добавить дополнительные шаги для Enterprise
            pass""",
    """        # Для Enterprise плана добавляем дополнительные шаги
        if self.plan == "enterprise":
            # Добавляем специальные шаги для Enterprise
            enterprise_steps = [
                OnboardingStep.ADVANCED_SETTINGS.value if hasattr(OnboardingStep, 'ADVANCED_SETTINGS') else 'advanced_settings',
                'compliance_setup',
                'custom_integrations'
            ]
            # Добавляем Enterprise шаги перед финальным шагом
            if OnboardingStep.COMPLETE.value in self.progress["next_steps"]:
                complete_index = self.progress["next_steps"].index(OnboardingStep.COMPLETE.value)
                for i, step in enumerate(enterprise_steps):
                    if step not in self.progress["next_steps"]:
                        self.progress["next_steps"].insert(complete_index + i, step)
            print(f"✅ Настроены специальные шаги для Enterprise плана: {len(enterprise_steps)} шагов")"""
)

with open("seo_ai_models/models/freemium/onboarding/onboarding_wizard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Заглушка pass в _configure_steps_for_plan исправлена!")
