# SEO AI Models - Полное руководство по установке и использованию

## 🎯 Что реализовано

### 1. AutoFix Engine - Автоматическое исправление SEO проблем
- ✅ Автоматический workflow: Analyze → Detect → Fix → Verify → Rollback
- ✅ 5 уровней сложности (TRIVIAL → CRITICAL)
- ✅ Backup/Rollback механизм
- ✅ 5 готовых fixers (meta tags, alt tags, content refresh, schema, internal links)

### 2. 10 Модулей улучшений SEO
1. **ContentRefreshAutomation** - авто-обновление устаревшего контента
2. **VisualContentAnalyzer** - оптимизация изображений + alt-теги
3. **IntentBasedOptimizer** - оптимизация под поисковые намерения
4. **CompetitorMonitor** - мониторинг конкурентов
5. **InternationalSEO** - мультиязычная оптимизация + hreflang
6. **LinkBuildingAssistant** - поиск возможностей для ссылок
7. **PredictiveAnalytics** - прогноз трафика и позиций
8. **CROIntegration** - оптимизация конверсии + A/B тесты
9. **MobileOptimizer** - Core Web Vitals + mobile-friendly
10. **AIContentGenerator** - генерация SEO-статей через LLM

### 3. Веб-интерфейс с real-time мониторингом
- ✅ **4 роли**: Admin, Analyst, User, Observer
- ✅ **JWT аутентификация** с session management
- ✅ **WebSocket** для live updates без перезагрузки
- ✅ **Dashboard** с активными анализами
- ✅ **Analysis Page** с real-time progress bar
- ✅ **Admin Panel** для управления пользователями
- ✅ **Mobile-responsive** дизайн

## 📦 Установка

### Backend

```bash
cd /home/user/seo-ai-models

# Установить зависимости (если еще не установлены)
pip install fastapi uvicorn sqlalchemy passlib[bcrypt] python-jose[cryptography] websockets

# Запустить сервер
uvicorn seo_ai_models.web.api.app:create_app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend

# Установить зависимости
npm install

# Запустить dev сервер
npm run dev
```

Фронтенд будет доступен на: **http://localhost:3000**
Backend API: **http://localhost:8000**
WebSocket: **ws://localhost:8000/ws**

## 👥 Создание первого пользователя (Admin)

### Вариант 1: Через Python скрипт

```python
from seo_ai_models.api.auth.service import AuthService
from seo_ai_models.api.auth.models import UserCreate, UserRole
from seo_ai_models.web.api.database.connection import SessionLocal

db = SessionLocal()

# Создать админа
admin_data = UserCreate(
    username="admin",
    email="admin@example.com",
    password="admin123",  # ИЗМЕНИТЕ В ПРОДАКШЕНЕ!
    role=UserRole.ADMIN,
    full_name="System Administrator"
)

admin_user = AuthService.create_user(db, admin_data)
print(f"Admin created: {admin_user.username}")

db.close()
```

### Вариант 2: Через API напрямую

```bash
curl -X POST "http://localhost:8000/api/v2/auth/users" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@example.com",
    "password": "admin123",
    "role": "admin",
    "full_name": "System Administrator"
  }'
```

## 🚀 Использование

### 1. Вход в систему

Откройте http://localhost:3000/login

**Логин:**
- Username: `admin`
- Password: `admin123`

### 2. Dashboard

После входа вы попадете на Dashboard, где можно:
- Видеть активные анализы в реальном времени
- Статистику (количество анализов, статус, роль)
- Запустить новый анализ кнопкой "+ New Analysis"

### 3. Запуск анализа

**Кнопка "New Analysis"** → заполнить форму:

```
URL: https://example.com
Content: [вставить текст страницы]
Keywords: seo, optimization, content
Enable Auto-Fix: ✅
Fix Complexity Limit: Simple
```

**Нажать "Start Analysis"**

### 4. Real-time мониторинг

После запуска анализа:
1. Вы будете перенаправлены на страницу анализа
2. **Progress bar** будет обновляться в реальном времени
3. Зеленая точка "Live" показывает WebSocket соединение
4. Шаги отображаются по мере выполнения:
   - "Running base SEO analysis" (20%)
   - "Analyzing visual content" (40%)
   - "Checking mobile friendliness" (60%)
   - "Applying automatic fixes" (80%)
   - "Complete" (100%)

5. После завершения видны результаты:
   - Overall Score: X/100
   - Issues Detected: N
   - Fixes Applied: M

### 5. Admin Panel (только для Admin)

**Menu → Admin Panel**

Функции:
- **Создать пользователя**: кнопка "+ Create User"
- **Редактировать**: иконка карандаша
- **Удалить**: иконка корзины
- **Изменить роль**: при редактировании
- **Активировать/деактивировать**: чекбокс "Active"

**Роли:**
- **Admin** - полный доступ + управление пользователями
- **Analyst** - запуск анализов + одобрение сложных autofixes
- **User** - запуск своих анализов + простые autofixes
- **Observer** - только просмотр (read-only)

## 🔧 Использование API напрямую

### Вход (получение JWT токена)

```bash
curl -X POST "http://localhost:8000/api/v2/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

Ответ:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "...",
    "username": "admin",
    "email": "admin@example.com",
    "role": "admin"
  }
}
```

### Запуск анализа

```bash
TOKEN="<your_jwt_token>"

curl -X POST "http://localhost:8000/api/v2/enhanced-analysis/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "content": "Your page content here...",
    "keywords": ["seo", "optimization"],
    "auto_fix": true,
    "fix_complexity_limit": "simple"
  }'
```

Ответ:
```json
{
  "analysis_id": "AbCdEf123456",
  "status": "started",
  "message": "Analysis started. Connect to WebSocket to track progress: /ws/analysis/AbCdEf123456"
}
```

### Проверка статуса

```bash
curl -X GET "http://localhost:8000/api/v2/enhanced-analysis/status/AbCdEf123456" \
  -H "Authorization: Bearer $TOKEN"
```

### Подключение к WebSocket (JavaScript)

```javascript
const token = "your_jwt_token";
const analysisId = "AbCdEf123456";

const ws = new WebSocket(`ws://localhost:8000/ws/analysis/${analysisId}?token=${token}`);

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'analysis_update') {
    console.log(`Progress: ${message.data.progress}%`);
    console.log(`Step: ${message.data.current_step}`);
  }
};
```

## 📊 Примеры использования

### Python: Полный цикл с EnhancedSEOAdvisor

```python
from seo_ai_models.models.enhanced_advisor import EnhancedSEOAdvisor

# Инициализация
advisor = EnhancedSEOAdvisor(auto_execute=True)

# Запуск анализа с auto-fix
result = advisor.analyze_and_fix(
    url="https://example.com",
    content="Your page content here...",
    keywords=["seo", "optimization", "content"],
    auto_fix=True,
    fix_complexity_limit=FixComplexity.SIMPLE
)

# Результаты
print(f"Overall Score: {result['overall_score']}/100")
print(f"Issues Detected: {len(result['issues_detected'])}")
print(f"Fixes Applied: {len(result['fixes_applied'])}")

for fix in result['fixes_applied']:
    print(f"- {fix['description']}")
```

### Только AutoFix без полного анализа

```python
from seo_ai_models.autofix.engine import AutoFixEngine, FixComplexity
from seo_ai_models.autofix.fixers import MetaTagsFixer

# Инициализация
engine = AutoFixEngine(auto_execute=True)

# Регистрация fixers
engine.register_action("missing_meta_tags", MetaTagsFixer(llm_service))

# Результаты анализа
analysis_results = {
    "missing_meta_tags": [
        {"page_id": "page1", "missing": ["title", "description"]}
    ]
}

# Создать план исправлений
plan = engine.analyze_and_plan(analysis_results)

# Выполнить (auto-execute для SIMPLE)
result = engine.execute_plan(plan, require_approval_for=[FixComplexity.COMPLEX])

print(f"Executed: {len(result['executed'])}")
print(f"Pending: {len(result['pending_approval'])}")
```

## 🔐 Безопасность

### Production Settings

В продакшене ОБЯЗАТЕЛЬНО изменить:

```python
# seo_ai_models/common/config/settings.py

settings = {
    "jwt_secret_key": "СГЕНЕРИРОВАТЬ_НОВЫЙ_СЕКРЕТНЫЙ_КЛЮЧ",  # ← ВАЖНО!
    "access_token_expire_minutes": 30,
    "allowed_origins": ["https://yourdomain.com"],  # ← Указать домен
}
```

### CORS настройка

```python
# seo_ai_models/web/api/app.py

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Не использовать "*" в продакшене!
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)
```

## 📖 API Документация

После запуска backend доступна Swagger UI:
**http://localhost:8000/docs**

Там можно:
- Увидеть все endpoints
- Попробовать API прямо в браузере
- Скачать OpenAPI спецификацию

## 🐛 Troubleshooting

### WebSocket не подключается

```bash
# Проверить что backend запущен
curl http://localhost:8000/api/health

# Проверить firewall/прокси не блокирует WebSocket
```

### "Invalid token" при запросах

```bash
# Токен истек (30 минут). Получить новый:
curl -X POST "http://localhost:8000/api/v2/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### Frontend не запускается

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Database ошибки

```bash
# Создать/обновить таблицы
python -c "from seo_ai_models.web.api.database.models import Base; from seo_ai_models.web.api.database.connection import engine; Base.metadata.create_all(bind=engine)"
```

## 📝 Дополнительная информация

### Структура проекта

```
seo-ai-models/
├── seo_ai_models/
│   ├── autofix/              # AutoFix Engine
│   │   ├── engine.py         # Ядро системы
│   │   └── fixers.py         # Готовые fixers
│   ├── improvements/         # 10 модулей улучшений
│   ├── models/
│   │   └── enhanced_advisor.py  # Главная интеграция
│   ├── api/                  # Новое API v2
│   │   ├── auth/            # Аутентификация
│   │   ├── routes/          # Endpoints
│   │   └── websocket/       # WebSocket
│   └── web/api/             # Старое API (совместимость)
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── pages/           # Страницы
│   │   ├── components/      # Компоненты
│   │   ├── stores/          # State management
│   │   ├── hooks/           # Custom hooks
│   │   └── lib/             # Утилиты
│   └── package.json
└── tests/                   # Тесты

```

### Commits

- `84e985f` - AutoFix Engine + 10 модулей улучшений
- `14cd3b2` - Веб-интерфейс + роли + WebSocket

### Контакты

Для вопросов и предложений создавайте Issues в репозитории.

---

**Готово!** Теперь у вас есть полноценная система SEO анализа с:
✅ Автоматическим исправлением проблем
✅ Real-time мониторингом через веб-интерфейс
✅ Ролевой системой доступа
✅ 10 модулями продвинутых улучшений
