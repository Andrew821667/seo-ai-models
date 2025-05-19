
from seo_ai_models.web.api.app import create_app

# Проверяем, что приложение создается без ошибок
app = create_app()
print("API application created successfully!")
print(f"Available routes:")
for route in app.routes:
    print(f" - {route.path} [{','.join(route.methods)}]")
