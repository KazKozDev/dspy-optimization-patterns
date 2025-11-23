# 🚀 Deployment Guide - Quick Start

## ⚠️ Current Environment
Docker не доступен в текущем окружении. Но проект полностью готов к запуску!

## 🎯 Варианты Запуска

### Вариант 1: На Локальной Машине (Рекомендуется)

#### Шаг 1: Клонируй репозиторий
```bash
git clone <your-repo-url>
cd dspy-optimization-patterns
```

#### Шаг 2: Настрой окружение
```bash
# Создай .env из шаблона
cp .env.example .env

# Добавь свой API ключ в .env
nano .env  # или vim, code, etc.
# OPENAI_API_KEY=sk-your-real-key-here
```

#### Шаг 3: Запусти через Docker Compose
```bash
# Поднять все сервисы (API + Qdrant + Phoenix)
docker-compose up -d

# Проверить статус
docker-compose ps

# Посмотреть логи
docker-compose logs -f api

# Остановить
docker-compose down
```

**Что запустится:**
- 🌐 API: http://localhost:8000
- 📚 Qdrant: http://localhost:6333
- 🔍 Phoenix: http://localhost:6006
- 📓 Swagger Docs: http://localhost:8000/docs

---

### Вариант 2: Без Docker (Python напрямую)

#### Установка
```bash
# Установить зависимости
pip install poetry
poetry install --with dev

# Или через pip
pip install -e .
```

#### Запуск компонентов

**1. Запустить Qdrant отдельно:**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**2. Запустить API:**
```bash
export OPENAI_API_KEY=sk-your-key
export STUDENT_MODEL=gpt-5o-mini

poetry run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Открыть в браузере:**
- http://localhost:8000/docs - API документация
- http://localhost:8000/health - Health check

---

### Вариант 3: Kubernetes (Production)

```bash
# Настроить секреты
kubectl create secret generic dspy-secrets \
  --from-literal=OPENAI_API_KEY=sk-xxx \
  -n dspy-production

# Развернуть
kubectl apply -f k8s/

# Проверить
kubectl get pods -n dspy-production
kubectl logs -f deployment/dspy-api -n dspy-production
```

---

## 🧪 Тестирование API

### Health Check
```bash
curl http://localhost:8000/health
```

### Question Answering
```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is DSPy?",
    "context": "DSPy is a framework for programming with foundation models."
  }'
```

### RAG Example
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does DSPy optimization work?",
    "top_k": 5
  }'
```

---

## 📊 Оптимизация Модели

### Подготовить данные
```bash
poetry run python scripts/prepare_data.py \
  --input data/raw/qa_dataset_sample.jsonl \
  --output data/processed/ \
  --task-type qa
```

### Запустить оптимизацию
```bash
# Быстрый способ
./scripts/optimize_model.sh SimpleRAG data/processed/qa_dataset.jsonl qa

# Или детально
poetry run python -m src.pipeline.optimizer \
  --module SimpleRAG \
  --data data/processed/qa_dataset.jsonl \
  --task-type qa \
  --metric hybrid_qa \
  --optimizer mipro \
  --output artifacts/compiled_programs/rag_v1.json
```

**Результат:** 
Скомпилированная программа сохранится в `artifacts/compiled_programs/`

---

## 🔍 Мониторинг

### Посмотреть логи
```bash
# Docker Compose
docker-compose logs -f api

# Kubernetes
kubectl logs -f deployment/dspy-api -n dspy-production

# Локальный файл
tail -f logs/dspy_*.jsonl
```

### Phoenix Observability
1. Запусти Phoenix: `docker-compose --profile observability up -d`
2. Открой: http://localhost:6006
3. Смотри trace всех запросов в реальном времени

---

## 🐛 Troubleshooting

### API не запускается
```bash
# Проверь что порт свободен
lsof -i :8000

# Проверь что .env настроен
cat .env | grep OPENAI_API_KEY

# Проверь логи
docker-compose logs api
```

### "Module not found" ошибка
```bash
# Переустанови зависимости
poetry install --with dev

# Или
pip install -e .
```

### Оптимизация слишком долгая
```bash
# Используй меньше данных
--train-size 20 --dev-size 50

# Или более простой оптимизатор
--optimizer bootstrap  # вместо mipro
```

---

## ✅ Готово!

Проект полностью настроен и готов к работе. Выбери любой вариант запуска выше.

**Нужна помощь?** Открой issue в GitHub или смотри README.md
