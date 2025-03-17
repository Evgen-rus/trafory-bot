"""
Модуль для обработки и векторизации базы знаний Trafory.
Предназначен для подготовки базы знаний к использованию в поисковой системе.
Включает функционал парсинга markdown-файлов, разбиения на чанки и создания векторных эмбеддингов.
"""
import os
import re
import pickle
import faiss
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import asyncio
from openai import AsyncOpenAI

from config import (
    OPENAI_API_KEY, 
    KB_DIR, 
    FAISS_INDEX_DIR, 
    FAISS_INDEX_PATH, 
    METADATA_PATH,
    EMBEDDING_MODEL, 
    EMBEDDING_DIMENSIONS,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Настраиваем логирование
logger = logging.getLogger(__name__)

class KnowledgeBaseProcessor:
    """
    Класс для обработки и векторизации базы знаний.
    Обеспечивает загрузку документов, их разбиение и векторизацию с помощью OpenAI API.
    """
    def __init__(self):
        """
        Инициализирует процессор базы знаний.
        
        Создает клиент OpenAI API с указанным ключом API.
        
        Исключения:
            ValueError: Если API ключ не указан или неверный.
        """
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.total_embedding_tokens = a = 0
        self.token_usage_by_file = {}
        
    async def create_embedding(self, text: str, file_path: str = None) -> Tuple[np.ndarray, int]:
        """
        Создает эмбеддинг (векторное представление) для текста с помощью OpenAI API.
        
        Вызывается при векторизации каждого чанка текста в методе vectorize_chunks.
        
        Args:
            text: Входной текст для векторизации
            file_path: Путь к файлу, для которого создается эмбеддинг (для статистики)
            
        Returns:
            Кортеж из вектора эмбеддинга и количества использованных токенов
            
        Исключения:
            openai.error.OpenAIError: При ошибке запроса к API OpenAI
            ValueError: Если текст пустой или слишком длинный
        """
        response = await self.client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        
        # Получаем количество использованных токенов
        tokens_used = response.usage.total_tokens
        
        # Обновляем общую статистику
        self.total_embedding_tokens += tokens_used
        
        # Если указан путь к файлу, обновляем статистику по файлам
        if file_path:
            if file_path not in self.token_usage_by_file:
                self.token_usage_by_file[file_path] = 0
            self.token_usage_by_file[file_path] += tokens_used
        
        logger.info(f"Создан эмбеддинг: использовано {tokens_used} токенов")
        
        return np.array(response.data[0].embedding, dtype=np.float32), tokens_used
    
    def load_markdown_files(self) -> List[Dict[str, Any]]:
        """
        Рекурсивно загружает все Markdown файлы из базы знаний.
        
        Вызывается в начале процесса обработки базы знаний в методе process_knowledge_base.
        
        Returns:
            Список словарей с содержимым файлов и метаданными:
            - title: заголовок документа
            - content: содержимое документа
            - path: относительный путь к файлу
            - file_name: имя файла
            - hierarchy: иерархия разделов
            
        Исключения:
            FileNotFoundError: Если директория базы знаний не существует
            UnicodeDecodeError: При ошибке декодирования файла
        """
        documents = []
        base_path = Path(KB_DIR)
        
        for file_path in base_path.glob('**/*.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # Извлекаем заголовок из первой строки (если это строка с #)
                    title = content.split('\n')[0].lstrip('#').strip() if content.startswith('#') else file_path.stem
                    
                    # Определяем путь относительно базовой директории
                    rel_path = str(file_path.relative_to(base_path))
                    
                    # Определяем иерархию документа на основе пути
                    hierarchy = rel_path.split(os.sep)
                    hierarchy = [h for h in hierarchy if h.endswith('.md') is False]
                    if file_path.stem != title:
                        hierarchy.append(title)
                    
                    documents.append({
                        'title': title,
                        'content': content,
                        'path': rel_path,
                        'file_name': file_path.name,
                        'hierarchy': hierarchy
                    })
                    print(f"Загружен файл: {rel_path}")
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")
        
        return documents
    
    def split_into_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Разбивает документы на смысловые чанки для векторизации.
        
        Вызывается после загрузки документов в методе process_knowledge_base.
        Разбивает тексты на части по заголовкам и абзацам с учетом максимального размера чанка.
        
        Args:
            documents: Список документов, каждый в виде словаря с метаданными
            
        Returns:
            Список чанков с метаданными, унаследованными от исходных документов и дополненными:
            - title: заголовок секции
            - content: содержимое чанка
            - parent_title: заголовок родительского документа
            
        Исключения:
            IndexError: При неправильной структуре документа
        """
        chunks = []
        
        for doc in documents:
            content = doc['content']
            
            # Разбиваем по заголовкам
            sections = re.split(r'(?=^#+ )', content, flags=re.MULTILINE)
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Получаем заголовок секции
                section_title = section.split('\n')[0].lstrip('#').strip() if section.startswith('#') else "Без заголовка"
                
                # Удаляем ссылки на изображения
                section = re.sub(r'!\[.*?\]\(.*?\)', '', section)
                
                # Примерная оценка размера в токенах (4 символа на токен)
                approx_tokens = len(section) // 4
                
                # Если секция слишком большая, разбиваем на абзацы
                if approx_tokens > MAX_CHUNK_SIZE:
                    paragraphs = re.split(r'\n\s*\n', section)
                    current_chunk = []
                    current_size = 0
                    
                    for paragraph in paragraphs:
                        paragraph_tokens = len(paragraph) // 4
                        
                        if current_size + paragraph_tokens > MAX_CHUNK_SIZE and current_chunk:
                            # Создаем новый чанк
                            chunk_content = '\n\n'.join(current_chunk)
                            chunks.append({
                                'title': section_title,
                                'content': chunk_content,
                                'path': doc['path'],
                                'file_name': doc['file_name'],
                                'parent_title': doc['title'],
                                'hierarchy': doc['hierarchy']
                            })
                            
                            # Берем перекрытие для следующего чанка (последний абзац текущего чанка)
                            if CHUNK_OVERLAP > 0 and len(current_chunk) > 0:
                                current_chunk = [current_chunk[-1]]
                                current_size = len(current_chunk[-1]) // 4
                            else:
                                current_chunk = []
                                current_size = 0
                                
                            current_chunk.append(paragraph)
                            current_size += paragraph_tokens
                        else:
                            current_chunk.append(paragraph)
                            current_size += paragraph_tokens
                    
                    # Добавляем последний сформированный чанк
                    if current_chunk:
                        chunk_content = '\n\n'.join(current_chunk)
                        chunks.append({
                            'title': section_title,
                            'content': chunk_content,
                            'path': doc['path'],
                            'file_name': doc['file_name'],
                            'parent_title': doc['title'],
                            'hierarchy': doc['hierarchy']
                        })
                else:
                    # Если секция помещается в один чанк
                    chunks.append({
                        'title': section_title,
                        'content': section,
                        'path': doc['path'],
                        'file_name': doc['file_name'],
                        'parent_title': doc['title'],
                        'hierarchy': doc['hierarchy']
                    })
        
        return chunks
    
    async def vectorize_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Векторизует все чанки и создает FAISS индекс.
        
        Вызывается после разбиения документов на чанки в методе process_knowledge_base.
        Для каждого чанка создается эмбеддинг, который добавляется в индекс FAISS.
        
        Args:
            chunks: Список чанков для векторизации с метаданными
            
        Returns:
            Кортеж из трех элементов:
            - индекс FAISS для поиска по косинусному расстоянию
            - список метаданных, соответствующих векторам в индексе
            - словарь со статистикой использования токенов
            
        Исключения:
            openai.error.OpenAIError: При ошибке запроса к API OpenAI
            ValueError: При проблемах с форматом данных
            RuntimeError: При ошибках в работе FAISS
        """
        # Создаем индекс FAISS для быстрого поиска по косинусному расстоянию
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        
        # Метаданные для каждого вектора
        metadata = []
        
        # Статистика по токенам
        token_stats = {
            "total_tokens": 0,
            "chunks_count": len(chunks),
            "tokens_by_file": {}
        }
        
        # Векторизуем все чанки
        print(f"Начинаем векторизацию {len(chunks)} чанков...")
        
        for i, chunk in enumerate(chunks):
            print(f"Векторизуем чанк {i+1}/{len(chunks)}: {chunk['title']}")
            
            # Очищаем текст от специальных символов Markdown
            cleaned_content = chunk['content']
            
            # Создаем эмбеддинг для содержимого чанка
            embedding, tokens_used = await self.create_embedding(cleaned_content, chunk['path'])
            
            # Обновляем статистику по токенам
            token_stats["total_tokens"] += tokens_used
            if chunk['path'] not in token_stats["tokens_by_file"]:
                token_stats["tokens_by_file"][chunk['path']] = 0
            token_stats["tokens_by_file"][chunk['path']] += tokens_used
            
            # Добавляем вектор в индекс
            index.add(np.array([embedding]))
            
            # Сохраняем метаданные (без самого вектора для экономии памяти)
            meta = {
                'title': chunk['title'],
                'content': cleaned_content,
                'path': chunk['path'],
                'file_name': chunk['file_name'],
                'parent_title': chunk['parent_title'],
                'hierarchy': chunk['hierarchy'],
                'tokens': tokens_used
            }
            metadata.append(meta)
        
        # Добавляем среднее количество токенов на чанк в статистику
        if len(chunks) > 0:
            token_stats["avg_tokens_per_chunk"] = token_stats["total_tokens"] / len(chunks)
        
        return index, metadata, token_stats
    
    def save_index(self, index: faiss.Index, metadata: List[Dict[str, Any]], token_stats: Dict[str, Any] = None) -> None:
        """
        Сохраняет индекс FAISS и метаданные на диск.
        
        Вызывается после создания индекса и метаданных в методе process_knowledge_base.
        
        Args:
            index: Индекс FAISS для сохранения
            metadata: Список метаданных для сохранения
            token_stats: Статистика использования токенов (опционально)
            
        Исключения:
            IOError: При ошибке записи файлов
            PermissionError: При отсутствии прав на запись
            pickle.PickleError: При ошибке сериализации метаданных
        """
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        
        # Сохраняем индекс FAISS
        faiss.write_index(index, FAISS_INDEX_PATH)
        
        # Сохраняем метаданные
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Сохраняем статистику токенов, если она предоставлена
        if token_stats:
            token_stats_path = os.path.join(FAISS_INDEX_DIR, "token_stats.pickle")
            with open(token_stats_path, 'wb') as f:
                pickle.dump(token_stats, f)
        
        print(f"Индекс и метаданные сохранены в {FAISS_INDEX_DIR}")
        
        if token_stats:
            print(f"Общее количество использованных токенов: {token_stats['total_tokens']}")
            if "avg_tokens_per_chunk" in token_stats:
                print(f"Среднее количество токенов на чанк: {token_stats['avg_tokens_per_chunk']:.2f}")
    
    async def process_knowledge_base(self) -> None:
        """
        Обрабатывает базу знаний: загружает файлы, разбивает на чанки и векторизует.
        
        Это основной метод класса, который последовательно вызывает все остальные методы для 
        полной подготовки базы знаний. Вызывается из точки входа в скрипт (функция main).
        
        Действия:
        1. Загружает все Markdown файлы из базы знаний
        2. Разбивает их на смысловые чанки
        3. Векторизует чанки с помощью OpenAI API
        4. Сохраняет индекс FAISS и метаданные на диск
        
        Исключения:
            Exception: Любые исключения, возникшие в процессе обработки
        """
        print(f"Начинаем обработку базы знаний из {KB_DIR}...")
        
        # Загружаем Markdown файлы
        documents = self.load_markdown_files()
        print(f"Загружено {len(documents)} документов")
        
        # Разбиваем на чанки
        chunks = self.split_into_chunks(documents)
        print(f"Создано {len(chunks)} чанков")
        
        # Векторизуем чанки
        index, metadata, token_stats = await self.vectorize_chunks(chunks)
        print(f"Векторизовано {len(metadata)} чанков")
        
        # Сохраняем индекс и метаданные
        self.save_index(index, metadata, token_stats)
        print("Обработка базы знаний завершена")
        
    def get_token_usage(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования токенов для создания эмбеддингов.
        
        Returns:
            Словарь с информацией об использованных токенах:
            - total_embedding_tokens: общее количество токенов для эмбеддингов
            - token_usage_by_file: распределение использования токенов по файлам
        """
        return {
            "total_embedding_tokens": self.total_embedding_tokens,
            "token_usage_by_file": self.token_usage_by_file
        }

async def main():
    """
    Основная функция для запуска обработки базы знаний.
    
    Точка входа при запуске скрипта kb_processor.py напрямую.
    Создает экземпляр KnowledgeBaseProcessor и запускает обработку базы знаний.
    
    Исключения:
        Exception: Любые исключения, возникшие в процессе обработки
    """
    processor = KnowledgeBaseProcessor()
    await processor.process_knowledge_base()

if __name__ == "__main__":
    # Настраиваем логирование для консоли
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    asyncio.run(main()) 