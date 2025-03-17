"""
Модуль для работы с OpenAI API.
Обеспечивает генерацию ответов на вопросы пользователей на основе контекста из базы знаний.
"""
import logging
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, COMPLETION_MODEL

# Настраиваем логирование
logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Класс для работы с OpenAI API.
    Используется для генерации ответов на вопросы пользователей.
    """
    def __init__(self):
        """
        Инициализирует сервис OpenAI.
        
        Создает клиент OpenAI API с указанным ключом API.
        
        Исключения:
            ValueError: Если API ключ не указан или неверный
        """
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
    
    async def generate_response(self, query: str, context: str, search_scores: List[float] = None) -> Tuple[str, Dict[str, int]]:
        """
        Генерирует ответ на основе запроса пользователя и контекста из базы знаний.
        
        Вызывается из обработчика сообщений пользователя в bot_handlers.py после
        получения результатов поиска в базе знаний.
        
        Args:
            query: Запрос пользователя
            context: Контекст из базы знаний (релевантные фрагменты)
            search_scores: Список оценок релевантности найденных фрагментов
            
        Returns:
            Кортеж из сгенерированного ответа и словаря с информацией о токенах:
            - prompt_tokens: количество токенов в запросе
            - completion_tokens: количество токенов в ответе
            - total_tokens: общее количество токенов
            
        Исключения:
            openai.error.OpenAIError: При ошибке запроса к API OpenAI
            ValueError: При проблемах с форматом данных
            RuntimeError: При неожиданных ошибках во время генерации ответа
        """
        # Проверяем и обрезаем контекст
        if len(context) > 2000:  # Примерное ограничение
            context = context[:2000] + "...\n[Контекст был обрезан из-за слишком большого размера]"
        
        # Проверка релевантности найденных результатов
        is_relevant_query = True
        relevance_threshold = 0.75  # Минимальный порог релевантности
        
        # Если переданы оценки релевантности и их среднее ниже порога
        if search_scores and len(search_scores) > 0:
            avg_relevance = sum(search_scores) / len(search_scores)
            is_relevant_query = avg_relevance >= relevance_threshold
        
        # Формируем промпт для модели
        prompt = f"""Ты — эксперт по платформе Trafory, отвечающий на вопросы пользователей.

Вопрос пользователя: {query}

Ниже представлена информация из базы знаний платформы Trafory, которая может помочь с ответом:

{context}

Правила ответа:
1. Используй ТОЛЬКО информацию из предоставленного контекста.
2. Если в контексте нет информации для полного ответа, честно скажи об этом.
3. Твой ответ должен быть структурированным, информативным и вежливым.
4. Ссылайся на разделы документации, упоминая название документа.
5. Используй дефисы и тире для списков вместо звездочек (* или **). НЕ ИСПОЛЬЗУЙ звездочки для выделения текста.
6. Отвечай на русском языке.
7. Будь эмпатичным и покажи готовность помочь пользователю.
8. Не придумывай информацию, которой нет в контексте.
9. ВАЖНО: Если запрос пользователя не относится к платформе Trafory или ее функциям, вежливо объясни, что ты специализируешься только на вопросах, связанных с Trafory, и не можешь ответить на вопросы по другим темам.
10. Не используй Markdown-форматирование со звездочками. Используй HTML-теги для форматирования, если это необходимо (<b></b> для жирного текста, <i></i> для курсива).

Твой ответ:"""
        
        # Добавляем информацию о нерелевантности, если запрос не соответствует теме
        system_prompt = "Ты — эксперт техподдержки по платформе Trafory. Отвечай на вопросы пользователей, опираясь только на предоставленную информацию из базы знаний. Не используй звездочки (* или **) для форматирования текста."
        
        if not is_relevant_query:
            system_prompt += " Если запрос не относится к платформе Trafory, ее функциям или документации, вежливо откажи, объяснив, что ты можешь отвечать только на вопросы о Trafory."
        
        # Отправляем запрос к OpenAI
        response = await self.client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1024
        )
        
        # Получаем информацию о токенах
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # Обновляем общую статистику
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        
        # Логируем использование токенов
        logger.info(f"Использовано токенов: промпт={prompt_tokens}, " 
                   f"ответ={completion_tokens}, всего={total_tokens}")
        
        # Создаем словарь с информацией о токенах
        token_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        # Получаем ответ и заменяем звездочки, если они всё же остались
        content = response.choices[0].message.content
        content = content.replace("**", "").replace("*", "")
        
        return content, token_info
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Возвращает статистику использования токенов.
        
        Returns:
            Словарь с информацией об общем количестве использованных токенов:
            - total_prompt_tokens: общее количество токенов в запросах
            - total_completion_tokens: общее количество токенов в ответах
            - total_tokens: общее количество токенов
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens
        } 