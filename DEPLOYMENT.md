# Production Deployment Guide

## System Requirements

- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- Docker & Docker Compose (optional)
- 2GB RAM minimum
- 10GB disk space

## Environment Variables

Create a `.env` file with the following variables:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/seo_ai_models

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password
CACHE_TTL=300

# JWT Authentication
SECRET_KEY=your-super-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# CORS Origins (comma-separated)
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

## Installation

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/your-org/seo-ai-models.git
cd seo-ai-models

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start services
docker-compose up -d

# 4. Run migrations
docker-compose exec api alembic upgrade head

# 5. Check health
curl http://localhost:8000/health
```

### Option 2: Manual Installation

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Setup database
createdb seo_ai_models
alembic upgrade head

# 3. Start Redis
redis-server

# 4. Start application
uvicorn seo_ai_models.web.api.app:app --host 0.0.0.0 --port 8000
```

## Database Migrations

### Create new migration

```bash
alembic revision --autogenerate -m "Add new feature"
```

### Apply migrations

```bash
alembic upgrade head
```

### Rollback

```bash
alembic downgrade -1
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at:
```
http://localhost:8000/metrics
```

Key metrics:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `cache_hits_total` - Cache hits
- `cache_misses_total` - Cache misses
- `db_queries_total` - Database queries
- `errors_total` - Application errors

### Cache Statistics

```bash
curl http://localhost:8000/api/cache/stats \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Cache health
curl http://localhost:8000/api/cache/health

# Metrics health
curl http://localhost:8000/metrics/health
```

## Performance Tuning

### Redis Configuration

```bash
# redis.conf
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
```

### PostgreSQL Configuration

```sql
-- For better performance
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
```

### Application Workers

```bash
# Use multiple workers for production
gunicorn seo_ai_models.web.api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 30
```

## Security

### HTTPS Setup (Nginx)

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /metrics {
        deny all;  # Restrict metrics endpoint
    }
}
```

### Rate Limiting

Install nginx rate limiting module or use application-level rate limiting.

## Backup Strategy

### Database Backup

```bash
# Daily backup
pg_dump seo_ai_models > backup_$(date +%Y%m%d).sql

# Restore
psql seo_ai_models < backup_20250115.sql
```

### Redis Backup

```bash
# Save snapshot
redis-cli SAVE

# Copy RDB file
cp /var/lib/redis/dump.rdb /backup/redis_$(date +%Y%m%d).rdb
```

## Scaling

### Horizontal Scaling

1. Deploy multiple API instances behind a load balancer
2. Use shared Redis and PostgreSQL
3. Enable session affinity if needed

### Database Replication

```sql
-- Setup read replicas for better read performance
-- Primary handles writes, replicas handle reads
```

## Troubleshooting

### High CPU Usage

```bash
# Check slow queries
SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;

# Check cache hit rate
curl http://localhost:8000/api/cache/stats
```

### Memory Issues

```bash
# Check Redis memory
redis-cli INFO memory

# Clear cache if needed
curl -X DELETE http://localhost:8000/api/cache/clear \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Connection Pool Exhausted

Increase database pool size in connection.py:
```python
engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Increase from default 5
    max_overflow=40
)
```

## Maintenance

### Routine Tasks

1. Monitor error logs daily
2. Review metrics weekly
3. Update dependencies monthly
4. Backup database daily
5. Test disaster recovery quarterly

### Log Rotation

```bash
# /etc/logrotate.d/seo-ai-models
/var/log/seo-ai-models/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 www-data www-data
}
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/seo-ai-models/issues
- Documentation: https://docs.yourdomain.com
- Email: support@yourdomain.com
