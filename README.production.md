# SEO AI Models - Production Deployment Guide

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/seo-ai-models.git
cd seo-ai-models

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env with your settings
nano .env

# 4. Start all services
docker-compose up -d

# 5. Check status
docker-compose ps

# 6. View logs
docker-compose logs -f api
```

### Manual Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r web_requirements.txt

# 2. Install Playwright browsers
python -m playwright install

# 3. Install the package
pip install -e .

# 4. Run the API
uvicorn seo_ai_models.web.api.app:app --host 0.0.0.0 --port 8000
```

## 📦 Project Structure

```
seo-ai-models/
├── seo_ai_models/              # Main package
│   ├── common/                 # Shared utilities
│   ├── models/                 # AI/ML models
│   ├── parsers/                # Web parsers
│   └── web/                    # Web API
│       ├── api/                # FastAPI application
│       │   ├── routers/        # API endpoints
│       │   └── services/       # Business logic
│       └── dashboard/          # Dashboard UI
├── tests/                      # Test suite
├── .github/workflows/          # CI/CD pipelines
└── docker-compose.yml          # Docker configuration
```

## 🔧 Configuration

### Environment Variables

See `.env.example` for all available options.

**Required:**
- `SECRET_KEY` - Application secret key
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string

**Optional:**
- `OPENAI_API_KEY` - For LLM features
- `SERPAPI_KEY` - For SERP analysis
- `WEBHOOK_SECRET_KEY` - For webhook security

### Database Setup

```bash
# Using Docker Compose (automatic)
docker-compose up -d postgres

# Manual setup
createdb seo_ai
python -m alembic upgrade head
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=seo_ai_models

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration
```

## 🔍 API Documentation

Once the API is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Projects
- `GET /projects/` - List projects
- `POST /projects/` - Create project
- `GET /projects/{id}` - Get project details
- `PUT /projects/{id}` - Update project
- `DELETE /projects/{id}` - Delete project (soft delete)

#### Tasks
- `GET /projects/{id}/tasks` - List tasks
- `POST /projects/{id}/tasks` - Create task
- `DELETE /projects/{id}/tasks/{task_id}` - Delete task

#### Webhooks
- `POST /webhooks/` - Create webhook
- `GET /webhooks/` - List webhooks
- `DELETE /webhooks/{id}` - Delete webhook

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linters
black seo_ai_models
flake8 seo_ai_models
mypy seo_ai_models
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

- **black** - Code formatting
- **flake8** - Linting
- **isort** - Import sorting
- **mypy** - Type checking

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Logs

```bash
# Docker Compose
docker-compose logs -f api

# Check all services
docker-compose logs -f
```

## 🐳 Docker Commands

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Execute command in container
docker-compose exec api python manage.py

# Rebuild and restart
docker-compose up -d --build
```

## 🔐 Security

### Best Practices

1. **Never commit secrets** - Use environment variables
2. **Use strong passwords** - Change default passwords in `.env`
3. **Enable HTTPS** - Use reverse proxy (nginx) with SSL
4. **Rate limiting** - Configure in application settings
5. **Regular updates** - Keep dependencies up to date

### Security Scan

```bash
# Run security scan
bandit -r seo_ai_models

# Check dependencies
safety check
```

## 📈 Performance

### Optimization Tips

1. **Use Redis caching** - For frequently accessed data
2. **Database indexing** - Add indexes for common queries
3. **Connection pooling** - Configure in database settings
4. **Async operations** - Use background workers for heavy tasks

### Scaling

```bash
# Scale API instances
docker-compose up -d --scale api=3

# Use load balancer (nginx)
# See docker-compose.prod.yml example
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📝 License

MIT License - see LICENSE file

## 🆘 Support

- **Documentation**: https://docs.seo-ai-models.com
- **Issues**: https://github.com/yourusername/seo-ai-models/issues
- **Discussions**: https://github.com/yourusername/seo-ai-models/discussions
