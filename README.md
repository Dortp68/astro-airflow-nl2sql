Важные замечания: "Проброс портов в docker compose override. ollama должен быть запущен на 0.0.0.0"
systemctl stop ollama #Если уже запущен сервис
export OLLAMA_HOST=0.0.0.0:11434
ollama serve

astro dev start && docker exec -it unsloth-fastapi bash -c "cd work && uvicorn app.main:app --host 0.0.0.0 --port 8000"