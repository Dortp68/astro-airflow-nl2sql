# NL2SQL Fine-Tuning Pipeline

## Описание проекта

Этот проект реализует пайплайн для fine-tuning больших языковых моделей (LLM) на задачу преобразования естественного языка в SQL-запросы (NL2SQL). Проект использует Unsloth для эффективного fine-tuning, MLflow для трекинга экспериментов и результатов, а также Apache Airflow для оркестрации всего процесса обучения. FastAPI приложение используется для запуска и управления процессом обучения, при этом само обучение и модели работают в отдельных контейнерах.

## Основные возможности

- **Fine-tuning LLM**: Поддержка supervised fine-tuning (SFT) и preference optimization (SimPO)
- **Генерация синтетических датасетов**: Возможность создания дополнительных тренировочных данных
- **Трекинг экспериментов**: Интеграция с MLflow для мониторинга метрик и результатов обучения
- **Оркестрация**: Полная автоматизация процесса обучения через Apache Airflow (через Astronomer.io)
- **API интерфейс**: FastAPI приложение для запуска обучения и мониторинга

## Технологии

- **Unsloth**: Для эффективного fine-tuning LLM
- **MLflow**: Трекинг экспериментов и метрик
- **Apache Airflow**: Оркестрация пайплайна обучения
- **FastAPI**: REST API для управления обучением
- **Ollama**: Для запуска и обслуживания моделей

## Требования

- Docker и Docker Compose
- Python 3.8+
- Ollama
- Git
- Astronomer CLI (для запуска Airflow)

## Запуск

### Запуск Ollama сервера

Важные замечания: "Проброс портов в docker compose override. ollama должен быть запущен на 0.0.0.0"

```bash
systemctl stop ollama  # Если уже запущен сервис
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

### Запуск проекта

Проект использует Astronomer для запуска Airflow и дополнительных сервисов. FastAPI приложение запускается в отдельном контейнере для управления процессами обучения.

```bash
astro dev start && docker exec -it unsloth-fastapi bash -c "cd work && uvicorn app.main:app --host 0.0.0.0 --port 8000"
```

Airflow DAG'и управляют оркестрацией, а FastAPI предоставляет API для запуска обучения в соответствующих контейнерах.

## Структура проекта

- `dags/`: Airflow DAG'и для оркестрации процессов обучения и генерации данных
- `src/`: Исходный код
  - `app/`: FastAPI приложение для управления обучением
    - `api/endpoints/`: API endpoints для обучения и мониторинга
- `tests/`: Тесты

## Использование

1. Подготовьте датасет для fine-tuning (через Airflow DAG'и или вручную)Ы
2. Настройте параметры обучения в конфигурационных файлах
3. Используйте FastAPI API для запуска обучения (SFT или SimPO)
4. Мониторьте прогресс обучения через MLflow UI и API endpoints мониторинга
5. Airflow DAG'и управляют генерацией синтетических датасетов и полным пайплайном

## API Endpoints

### Мониторинг
- `GET /monitoring/ready`: Проверка готовности сервиса
- `GET /monitoring/health`: Проверка здоровья сервиса
- `GET /monitoring/ping`: Пинг сервиса
- `GET /monitoring/status`: Общий статус

### Обучение
- `POST /train/sft`: Запуск supervised fine-tuning
- `POST /train/simpo`: Запуск preference optimization (SimPO)
- `POST /train/stop`: Остановка текущего обучения
