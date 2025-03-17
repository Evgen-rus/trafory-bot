"""
Модуль с обработчиками для телеграм-бота.
Содержит все обработчики сообщений и команд от пользователей.
"""
import logging
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext

from kb_search import KnowledgeBaseSearch
from openai_service import OpenAIService

# Настраиваем логирование
logger = logging.getLogger(__name__)

# Создаем роутер для обработчиков
router = Router()

@router.message(Command("start"))
async def command_start(message: Message) -> None:
    """
    Обработчик команды /start.
    
    Вызывается когда пользователь отправляет команду /start или впервые запускает бота.
    Отправляет приветственное сообщение с описанием функционала бота.
    
    Args:
        message: Объект сообщения от пользователя
        
    Исключения:
        TelegramAPIError: При ошибке отправки сообщения
    """
    await message.answer(
        "👋 Здравствуйте! Я бот-помощник по платформе Trafory.\n\n"
        "Задайте мне вопрос о функциях, настройках или использовании платформы, "
        "и я найду ответ в базе знаний Trafory."
    )

@router.message(Command("help"))
async def command_help(message: Message) -> None:
    """
    Обработчик команды /help.
    
    Вызывается когда пользователь отправляет команду /help.
    Отправляет справочное сообщение с примерами запросов.
    
    Args:
        message: Объект сообщения от пользователя
        
    Исключения:
        TelegramAPIError: При ошибке отправки сообщения
    """
    await message.answer(
        "🔍 Я могу помочь вам найти информацию о платформе Trafory.\n\n"
        "Просто задайте вопрос в чате, например:\n"
        "- Как настроить орг.структуру компании?\n"
        "- Что такое штатная единица?\n"
        "- Как добавить нового пользователя?\n\n"
        "Я поищу ответ в базе знаний и предоставлю вам самую релевантную информацию."
    )

@router.message(Command("about"))
async def command_about(message: Message) -> None:
    """
    Обработчик команды /about.
    
    Вызывается когда пользователь отправляет команду /about.
    Отправляет информацию о боте, его возможностях и особенностях.
    
    Args:
        message: Объект сообщения от пользователя
        
    Исключения:
        TelegramAPIError: При ошибке отправки сообщения
    """
    await message.answer(
        "<b>🤖 О боте Trafory</b>\n\n"
        "Я - специализированный ассистент по платформе Trafory, разработанный для оперативного предоставления точной информации из базы знаний.\n\n"
        "<b>Мои возможности:</b>\n"
        "• Поиск ответов в базе знаний Trafory\n"
        "• Предоставление точной и актуальной информации о функциях платформы\n"
        "• Обработка запросов на естественном языке\n"
        "• Отслеживание использования токенов и оптимизация затрат\n\n"
        "<b>Технологии:</b>\n"
        "• Векторное хранилище FAISS для быстрого семантического поиска\n"
        "• Модели OpenAI для генерации ответов\n"
        "• Асинхронная архитектура для быстрого отклика\n\n"
        "<i>Версия: 1.0.0</i>"
    )

@router.message(Command("stats"))
async def command_stats(message: Message, kb_search: KnowledgeBaseSearch, 
                         openai_service: OpenAIService) -> None:
    """
    Обработчик команды /stats.
    
    Вызывается когда пользователь отправляет команду /stats.
    Отправляет статистику использования токенов.
    
    Args:
        message: Объект сообщения от пользователя
        kb_search: Экземпляр класса KnowledgeBaseSearch для получения статистики
        openai_service: Экземпляр класса OpenAIService для получения статистики
        
    Исключения:
        TelegramAPIError: При ошибке отправки сообщения
    """
    # Получаем статистику токенов
    openai_stats = openai_service.get_token_usage()
    search_stats = kb_search.get_token_usage()
    embedding_stats = kb_search.kb_processor.get_token_usage() if hasattr(kb_search, 'kb_processor') else {}
    
    # Формируем сообщение со статистикой
    stats_message = "📊 <b>Статистика использования API OpenAI</b>\n\n"
    
    stats_message += "<b>Генерация ответов:</b>\n"
    stats_message += f"• Токенов промптов: {openai_stats.get('total_prompt_tokens', 0)}\n"
    stats_message += f"• Токенов ответов: {openai_stats.get('total_completion_tokens', 0)}\n"
    stats_message += f"• Всего токенов: {openai_stats.get('total_tokens', 0)}\n\n"
    
    stats_message += "<b>Поисковые запросы:</b>\n"
    stats_message += f"• Всего запросов: {search_stats.get('search_queries_count', 0)}\n"
    stats_message += f"• Всего токенов: {search_stats.get('total_search_tokens', 0)}\n"
    
    if search_stats.get('search_queries_count', 0) > 0:
        stats_message += f"• Среднее на запрос: {search_stats.get('avg_tokens_per_search', 0):.1f}\n\n"
    else:
        stats_message += "• Среднее на запрос: 0\n\n"
    
    stats_message += "<b>Эмбеддинги базы знаний:</b>\n"
    stats_message += f"• Всего токенов: {embedding_stats.get('total_embedding_tokens', 0)}\n\n"
    
    # Общая статистика
    total_all = (openai_stats.get('total_tokens', 0) + 
                search_stats.get('total_search_tokens', 0) + 
                embedding_stats.get('total_embedding_tokens', 0))
    
    stats_message += f"<b>ИТОГО использовано токенов:</b> {total_all}"
    
    await message.answer(stats_message, parse_mode="HTML")

@router.message(F.text)
async def handle_message(message: Message, kb_search: KnowledgeBaseSearch, 
                        openai_service: OpenAIService) -> None:
    """
    Обработчик текстовых сообщений пользователя.
    
    Вызывается когда пользователь отправляет любое текстовое сообщение.
    Выполняет поиск по базе знаний и генерацию ответа с помощью OpenAI.
    
    Args:
        message: Объект сообщения от пользователя
        kb_search: Экземпляр класса KnowledgeBaseSearch для поиска в базе знаний
        openai_service: Экземпляр класса OpenAIService для генерации ответов
        
    Действия:
        1. Отправляет сообщение о начале обработки запроса
        2. Ищет релевантные фрагменты в базе знаний
        3. Формирует контекст для модели
        4. Генерирует ответ с помощью GPT
        5. Отправляет ответ пользователю
        
    Исключения:
        TelegramAPIError: При ошибке отправки сообщения
        ValueError: При проблемах с форматом данных
        RuntimeError: При неожиданных ошибках в процессе обработки запроса
    """
    query = message.text
    
    # Отправляем сообщение о том, что бот обрабатывает запрос
    processing_msg = await message.answer("🔍 Ищу информацию, пожалуйста, подождите...")
    
    try:
        # Ищем релевантные чанки в базе знаний
        search_results, search_tokens, relevance_scores = await kb_search.search(query, top_k=5)
        
        # Логируем использование токенов
        logger.info(f"Поиск по запросу '{query}': использовано {search_tokens} токенов")
        
        # Проверка на полное отсутствие результатов или крайне низкую релевантность
        if not search_results or (relevance_scores and max(relevance_scores) < 0.6):
            await processing_msg.edit_text(
                "Ваш вопрос, похоже, не относится к платформе Trafory или нашей базе знаний. "
                "Я специализируюсь только на вопросах, связанных с платформой Trafory. "
                "Пожалуйста, задайте вопрос о функциях, настройках или использовании платформы Trafory."
            )
            return
        
        # Форматируем найденные чанки в контекст для модели
        context = kb_search.format_context_from_results(search_results)
        
        # Генерируем ответ с помощью GPT, передавая оценки релевантности
        response, token_info = await openai_service.generate_response(query, context, relevance_scores)
        
        # Логируем использование токенов для генерации ответа
        logger.info(f"Генерация ответа: использовано {token_info['total_tokens']} токенов")
        
        # Отправляем ответ пользователю
        await processing_msg.edit_text(response)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
        await processing_msg.edit_text(
            "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
        ) 