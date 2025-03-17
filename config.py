"""
Конфигурационный файл для телеграм-бота базы знаний Trafory.
Содержит все настройки и параметры проекта, включая пути к файлам,
API-ключи и параметры обработки данных.
"""
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
# Это позволяет хранить секретные ключи отдельно от кода
load_dotenv()

# Telegram Bot API Token - используется для авторизации бота в Telegram API
# Получается из .env файла
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# OpenAI API Key - используется для доступа к API OpenAI
# Получается из .env файла
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Пути к файлам и директориям
# BASE_DIR - корневая директория проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# KB_DIR - путь к базе знаний Trafory
KB_DIR = os.path.join(os.path.dirname(BASE_DIR), "Trafory", "База знаний платформы Trafory")
# FAISS_INDEX_DIR - директория для хранения индекса FAISS
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")
# FAISS_INDEX_PATH - путь к бинарному файлу индекса FAISS
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "faiss_index.bin")
# METADATA_PATH - путь к файлу метаданных для индекса
METADATA_PATH = os.path.join(FAISS_INDEX_DIR, "metadata.pickle")

# Настройки OpenAI
# EMBEDDING_MODEL - модель для создания эмбеддингов (векторных представлений текста)
# text-embedding-ada-002 хорошо работает с русским языком
EMBEDDING_MODEL = "text-embedding-ada-002"
# COMPLETION_MODEL - модель для генерации ответов
COMPLETION_MODEL = "gpt-4o-mini"
# EMBEDDING_DIMENSIONS - размерность векторов для модели text-embedding-ada-002
EMBEDDING_DIMENSIONS = 1536

# Настройки чанкинга (разбиения текста на части)
# MAX_CHUNK_SIZE - максимальный размер чанка в токенах (приблизительная оценка)
MAX_CHUNK_SIZE = 800
# CHUNK_OVERLAP - перекрытие между соседними чанками в токенах
CHUNK_OVERLAP = 50 