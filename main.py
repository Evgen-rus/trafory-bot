"""
Главный файл для запуска телеграм-бота.
Точка входа в приложение, содержит инициализацию и запуск бота.
"""
import asyncio
import logging
import sys
import json
from datetime import datetime
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.types import BotCommand, BotCommandScopeDefault

from config import TELEGRAM_TOKEN
from bot_handlers import router
from kb_search import KnowledgeBaseSearch
from openai_service import OpenAIService

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Создаем экземпляры классов, используемых в обработчиках
kb_search = KnowledgeBaseSearch()
openai_service = OpenAIService()

# Создаем класс middleware для передачи сервисов в обработчики
class ServicesMiddleware(BaseMiddleware):
    def __init__(self, kb_search, openai_service):
        self.kb_search = kb_search
        self.openai_service = openai_service
        super().__init__()

    async def __call__(self, handler, event, data):
        # Добавляем сервисы в контекст данных
        data["kb_search"] = self.kb_search
        data["openai_service"] = self.openai_service
        
        # Важно вернуть результат вызова обработчика
        return await handler(event, data)

async def set_commands(bot: Bot):
    """
    Устанавливает команды для меню бота.
    
    Добавляет команды в меню бота, которые будут доступны пользователю.
    
    Args:
        bot: Экземпляр бота
    """
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="help", description="Помощь по использованию"),
        BotCommand(command="stats", description="Статистика использования"),
        BotCommand(command="about", description="О боте Trafory")
    ]
    
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())
    logging.info("Команды бота настроены")

async def log_token_usage(kb_search: KnowledgeBaseSearch, openai_service: OpenAIService) -> None:
    """
    Асинхронная функция для периодического логирования статистики использования токенов.
    
    Каждый час собирает статистику использования токенов из сервисов OpenAI и поиска
    по базе знаний, логирует ее и сохраняет в JSON-файл.
    
    Args:
        kb_search: Экземпляр класса KnowledgeBaseSearch для получения статистики эмбеддингов
        openai_service: Экземпляр класса OpenAIService для получения статистики генерации ответов
        
    Исключения:
        Exception: Любые исключения, возникшие при логировании статистики
    """
    while True:
        try:
            # Получаем статистику токенов из обоих сервисов
            openai_tokens = openai_service.get_token_usage()
            kb_tokens = kb_search.kb_processor.get_token_usage() if hasattr(kb_search, 'kb_processor') else {}
            
            # Объединяем статистику в один словарь
            combined_stats = {
                "timestamp": datetime.now().isoformat(),
                "completion": {
                    "prompt_tokens": openai_tokens.get("total_prompt_tokens", 0),
                    "completion_tokens": openai_tokens.get("total_completion_tokens", 0),
                    "total_tokens": openai_tokens.get("total_tokens", 0)
                },
                "embedding": {
                    "total_tokens": kb_tokens.get("total_embedding_tokens", 0)
                },
                "grand_total": openai_tokens.get("total_tokens", 0) + kb_tokens.get("total_embedding_tokens", 0)
            }
            
            # Логируем статистику
            logging.info(f"Статистика использования токенов: {json.dumps(combined_stats, ensure_ascii=False, indent=2)}")
            
            # Сохраняем статистику в файл
            try:
                with open("token_usage_stats.json", "w", encoding="utf-8") as f:
                    json.dump(combined_stats, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.error(f"Ошибка при сохранении статистики в файл: {e}")
            
        except Exception as e:
            logging.error(f"Ошибка при сборе статистики токенов: {e}")
        
        # Ждем 1 час перед следующим логированием
        await asyncio.sleep(3600)  # 3600 секунд = 1 час

async def main() -> None:
    """
    Основная функция для запуска бота.
    
    Точка входа при запуске приложения. Инициализирует бота, диспетчер и
    подключает обработчики сообщений.
    
    Действия:
        1. Инициализирует бота с токеном из конфигурации
        2. Создает диспетчер с хранилищем состояний
        3. Подключает роутер с обработчиками
        4. Регистрирует middleware для передачи объектов в обработчики
        5. Запускает задачу логирования статистики в фоновом режиме
        6. Запускает бота в режиме long polling
        
    Исключения:
        Exception: Любые исключения, возникшие при запуске бота
    """
    # Инициализируем бота и диспетчер с новым синтаксисом
    bot = Bot(
        token=TELEGRAM_TOKEN, 
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher(storage=MemoryStorage())
    
    # Подключаем роутер с обработчиками
    dp.include_router(router)
    
    # Регистрируем middleware данных для передачи объектов в обработчики
    dp.message.middleware.register(ServicesMiddleware(kb_search, openai_service))
    
    # Запускаем задачу периодического логирования статистики токенов в фоновом режиме
    token_logger_task = asyncio.create_task(log_token_usage(kb_search, openai_service))
    
    # Устанавливаем команды бота
    await set_commands(bot)
    
    # Запускаем бота
    logging.info("Запуск бота...")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        # При завершении работы бота, отменяем задачу логирования
        token_logger_task.cancel()
        # Логируем финальную статистику
        openai_tokens = openai_service.get_token_usage()
        kb_tokens = kb_search.kb_processor.get_token_usage() if hasattr(kb_search, 'kb_processor') else {}
        logging.info(f"Итоговая статистика по токенам: Генерация ответов - {openai_tokens.get('total_tokens', 0)}, "
                   f"Эмбеддинги - {kb_tokens.get('total_embedding_tokens', 0)}")

if __name__ == "__main__":
    asyncio.run(main()) 