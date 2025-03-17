"""
Модуль для поиска по векторизованной базе знаний Trafory.
Обеспечивает семантический поиск по базе знаний на основе векторных представлений текста.
"""
import os
import faiss
import pickle
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI

from config import (
    OPENAI_API_KEY,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL
)
from kb_processor import KnowledgeBaseProcessor

# Настраиваем логирование
logger = logging.getLogger(__name__)

class KnowledgeBaseSearch:
    """
    Класс для поиска по векторной базе знаний.
    Обеспечивает семантический поиск по базе знаний с использованием FAISS и OpenAI.
    """
    def __init__(self):
        """
        Инициализирует поиск по базе знаний.
        
        Создает клиент OpenAI API и загружает индекс FAISS с метаданными.
        Также инициализирует счетчики использования токенов для эмбеддингов.
        
        Исключения:
            ValueError: Если API ключ не указан или неверный
            FileNotFoundError: Если файлы индекса или метаданных не найдены
        """
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Загружаем индекс и метаданные
        self.index = None
        self.metadata = None
        self.load_index()
        
        # Инициализируем процессор базы знаний для доступа к методам подсчета токенов
        self.kb_processor = KnowledgeBaseProcessor()
        
        # Счетчики токенов для поисковых запросов
        self.total_search_tokens = 0
        self.search_queries_count = 0
        
    def load_index(self) -> None:
        """
        Загружает индекс FAISS и метаданные с диска.
        
        Вызывается при инициализации класса KnowledgeBaseSearch.
        Проверяет наличие файлов индекса и метаданных и загружает их.
        
        Исключения:
            FileNotFoundError: Если файлы индекса или метаданных не найдены
            pickle.UnpicklingError: При ошибке десериализации метаданных
            IOError: При ошибке чтения файлов
        """
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            
            with open(METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
                
            print(f"Индекс и метаданные загружены: {len(self.metadata)} чанков")
            
            # Загружаем статистику токенов, если она доступна
            token_stats_path = os.path.join(os.path.dirname(FAISS_INDEX_PATH), "token_stats.pickle")
            if os.path.exists(token_stats_path):
                with open(token_stats_path, 'rb') as f:
                    token_stats = pickle.load(f)
                    print(f"Статистика токенов загружена: {token_stats.get('total_tokens', 0)} токенов использовано при векторизации")
        else:
            print("Индекс или метаданные не найдены. Сначала запустите векторизацию базы знаний.")
            self.index = None
            self.metadata = None
    
    async def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], int, List[float]]:
        """
        Ищет наиболее релевантные чанки по запросу пользователя.
        
        Вызывается из обработчика сообщений пользователя в bot_handlers.py.
        Создает эмбеддинг для запроса и ищет ближайшие векторы в индексе FAISS.
        
        Args:
            query: Запрос пользователя
            top_k: Количество результатов для возврата
            
        Returns:
            Кортеж из трех элементов:
            - список наиболее релевантных чанков с метаданными и оценкой релевантности
            - количество использованных токенов
            - список оценок релевантности (косинусное сходство) для последующего анализа
            
        Исключения:
            ValueError: Если индекс не загружен или при проблемах с форматом данных
            openai.error.OpenAIError: При ошибке запроса к API OpenAI
            RuntimeError: При ошибках в работе FAISS
        """
        if self.index is None or self.metadata is None:
            raise ValueError("Индекс не загружен. Запустите векторизацию базы знаний.")
        
        # Создаем эмбеддинг для запроса
        response = await self.client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL
        )
        query_vector = np.array([response.data[0].embedding], dtype=np.float32)
        
        # Учитываем использованные токены
        tokens_used = response.usage.total_tokens
        self.total_search_tokens += tokens_used
        self.search_queries_count += 1
        self.kb_processor.total_embedding_tokens += tokens_used
        
        logger.info(f"Поисковый запрос: использовано {tokens_used} токенов")
        
        # Ищем ближайшие векторы
        scores, indices = self.index.search(query_vector, top_k)
        
        # Формируем результаты
        results = []
        relevance_scores = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Проверяем, что индекс валидный
                # Нормализуем косинусное расстояние в косинусное сходство (1 - лучшее, 0 - худшее)
                # FAISS возвращает отрицательное косинусное расстояние, поэтому добавляем 1
                similarity = float(1 + scores[0][i])
                relevance_scores.append(similarity)
                
                result = self.metadata[idx].copy()
                result['score'] = similarity
                results.append(result)
        
        return results, tokens_used, relevance_scores
    
    def format_context_from_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Форматирует результаты поиска в контекст для модели генерации ответов.
        
        Вызывается после получения результатов поиска в обработчике сообщений.
        
        Args:
            results: Результаты поиска от метода search
            
        Returns:
            Отформатированный контекст для передачи модели GPT
            
        Исключения:
            KeyError: При отсутствии необходимых полей в результатах
        """
        if not results:
            return "Информации по запросу не найдено."
        
        context_parts = []
        
        for i, result in enumerate(results):
            # Формируем иерархию заголовков
            hierarchy_text = " > ".join(result['hierarchy'])
            
            # Форматируем кусок контекста
            context_part = f"### Фрагмент {i+1}: {result['title']}\n"
            context_part += f"**Источник:** {hierarchy_text}\n\n"
            context_part += result['content']
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def get_token_usage(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования токенов для поисковых запросов.
        
        Returns:
            Словарь с информацией об использованных токенах:
            - total_search_tokens: общее количество токенов для поисковых запросов
            - search_queries_count: количество выполненных поисковых запросов
            - avg_tokens_per_search: среднее количество токенов на запрос
        """
        avg_tokens = 0
        if self.search_queries_count > 0:
            avg_tokens = self.total_search_tokens / self.search_queries_count
            
        return {
            "total_search_tokens": self.total_search_tokens,
            "search_queries_count": self.search_queries_count,
            "avg_tokens_per_search": avg_tokens
        } 