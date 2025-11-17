"""
Интеллектуальное кэширование результатов LLM-анализа.

Модуль предоставляет функционал для умного кэширования результатов анализа
LLM-моделей, с учетом контекста запроса, потенциальной выгоды кэширования
и интеллектуального управления временем жизни кэша.
"""

import os
import json
import hashlib
import logging
import time
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Импортируем необходимые компоненты
from ..service.cost_estimator import CostEstimator
from ..common.utils import parse_json_response


class IntelligentCache:
    """
    Интеллектуальное кэширование результатов LLM-анализа.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size_mb: int = 500,
        default_ttl: int = 86400 * 7,
    ):  # 7 дней
        """
        Инициализирует интеллектуальный кэш.

        Args:
            cache_dir: Директория для хранения кэша (опционально)
            max_cache_size_mb: Максимальный размер кэша в МБ
            default_ttl: Время жизни кэша по умолчанию в секундах
        """
        # Настройка директории кэша
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/seo_ai_models/intelligent_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Создаем директории для разных типов кэша
        self.analysis_cache_dir = os.path.join(self.cache_dir, "analysis")
        self.query_cache_dir = os.path.join(self.cache_dir, "query")
        self.metadata_cache_dir = os.path.join(self.cache_dir, "metadata")

        os.makedirs(self.analysis_cache_dir, exist_ok=True)
        os.makedirs(self.query_cache_dir, exist_ok=True)
        os.makedirs(self.metadata_cache_dir, exist_ok=True)

        # Настройка параметров кэша
        self.max_cache_size_mb = max_cache_size_mb
        self.default_ttl = default_ttl

        # Метаданные кэша
        self.cache_metadata = {
            "last_cleanup": datetime.now().isoformat(),
            "total_cache_hits": 0,
            "total_cache_misses": 0,
            "total_cost_saved": 0.0,
            "cache_entries": {},
        }

        # Настройка логгирования
        self.logger = logging.getLogger(__name__)

        # Загружаем метаданные кэша
        self._load_cache_metadata()

        # Проверяем размер кэша и очищаем его при необходимости
        self._check_cache_size()

    def _load_cache_metadata(self) -> None:
        """
        Загружает метаданные кэша.
        """
        metadata_file = os.path.join(self.metadata_cache_dir, "cache_metadata.json")

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.cache_metadata = json.load(f)
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке метаданных кэша: {str(e)}")

    def _save_cache_metadata(self) -> None:
        """
        Сохраняет метаданные кэша.
        """
        metadata_file = os.path.join(self.metadata_cache_dir, "cache_metadata.json")

        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении метаданных кэша: {str(e)}")

    def _check_cache_size(self) -> None:
        """
        Проверяет размер кэша и очищает его при необходимости.
        """
        # Получаем размер кэша
        cache_size = self._get_directory_size(self.cache_dir) / (1024 * 1024)  # в МБ

        # Если размер кэша превышает максимальный, выполняем очистку
        if cache_size > self.max_cache_size_mb:
            self.logger.info(
                f"Размер кэша ({cache_size:.2f} МБ) превышает максимальный ({self.max_cache_size_mb} МБ), выполняем очистку"
            )
            self._cleanup_cache()

    def _get_directory_size(self, path: str) -> int:
        """
        Получает размер директории в байтах.

        Args:
            path: Путь к директории

        Returns:
            int: Размер директории в байтах
        """
        total_size = 0

        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)

        return total_size

    def _cleanup_cache(self) -> None:
        """
        Очищает кэш, удаляя устаревшие и наименее полезные записи.
        """
        # Время последней очистки
        last_cleanup = datetime.fromisoformat(
            self.cache_metadata.get("last_cleanup", datetime.now().isoformat())
        )

        # Если с момента последней очистки прошло меньше часа, пропускаем
        if datetime.now() - last_cleanup < timedelta(hours=1):
            return

        self.logger.info("Очистка кэша...")

        # Обновляем время последней очистки
        self.cache_metadata["last_cleanup"] = datetime.now().isoformat()

        # Получаем список файлов в кэше с их метаданными
        cache_files = []

        # Анализируем файлы в analysis_cache_dir
        for file_name in os.listdir(self.analysis_cache_dir):
            file_path = os.path.join(self.analysis_cache_dir, file_name)

            if os.path.isfile(file_path):
                # Получаем метаданные файла
                cache_key = os.path.splitext(file_name)[0]
                cache_entry = self.cache_metadata.get("cache_entries", {}).get(cache_key, {})

                # Получаем время создания, полезность и размер
                creation_time = cache_entry.get(
                    "creation_time", datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                )
                usefulness = cache_entry.get("usefulness", 0.0)
                size = os.path.getsize(file_path)

                cache_files.append(
                    {
                        "file_path": file_path,
                        "cache_key": cache_key,
                        "type": "analysis",
                        "creation_time": creation_time,
                        "usefulness": usefulness,
                        "size": size,
                    }
                )

        # Анализируем файлы в query_cache_dir
        for file_name in os.listdir(self.query_cache_dir):
            file_path = os.path.join(self.query_cache_dir, file_name)

            if os.path.isfile(file_path):
                # Получаем метаданные файла
                cache_key = os.path.splitext(file_name)[0]
                cache_entry = self.cache_metadata.get("cache_entries", {}).get(cache_key, {})

                # Получаем время создания, полезность и размер
                creation_time = cache_entry.get(
                    "creation_time", datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                )
                usefulness = cache_entry.get("usefulness", 0.0)
                size = os.path.getsize(file_path)

                cache_files.append(
                    {
                        "file_path": file_path,
                        "cache_key": cache_key,
                        "type": "query",
                        "creation_time": creation_time,
                        "usefulness": usefulness,
                        "size": size,
                    }
                )

        # Сортируем файлы по полезности (от низкой к высокой)
        cache_files.sort(key=lambda x: x["usefulness"])

        # Удаляем файлы, начиная с наименее полезных, пока размер кэша не станет достаточно малым
        current_size = self._get_directory_size(self.cache_dir) / (1024 * 1024)  # в МБ
        target_size = self.max_cache_size_mb * 0.8  # Целевой размер после очистки

        for file_info in cache_files:
            # Если текущий размер меньше целевого, останавливаемся
            if current_size <= target_size:
                break

            # Удаляем файл
            try:
                os.remove(file_info["file_path"])

                # Обновляем текущий размер
                current_size -= file_info["size"] / (1024 * 1024)

                # Удаляем запись из метаданных
                if file_info["cache_key"] in self.cache_metadata.get("cache_entries", {}):
                    del self.cache_metadata["cache_entries"][file_info["cache_key"]]

                self.logger.info(f"Удален файл кэша: {file_info['file_path']}")
            except Exception as e:
                self.logger.error(
                    f"Ошибка при удалении файла кэша {file_info['file_path']}: {str(e)}"
                )

        # Сохраняем обновленные метаданные
        self._save_cache_metadata()

        self.logger.info(f"Очистка кэша завершена. Текущий размер: {current_size:.2f} МБ")

    def get_analysis_from_cache(
        self, content: str, analysis_type: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Получает результат анализа из кэша.

        Args:
            content: Контент для анализа
            analysis_type: Тип анализа
            params: Дополнительные параметры анализа (опционально)

        Returns:
            Optional[Dict[str, Any]]: Результат анализа из кэша или None, если не найден
        """
        # Создаем ключ кэша
        cache_key = self._create_cache_key(content, analysis_type, params)

        # Путь к файлу кэша
        cache_file = os.path.join(self.analysis_cache_dir, f"{cache_key}.json")

        # Проверяем наличие файла кэша
        if not os.path.exists(cache_file):
            # Увеличиваем счетчик кэш-миссов
            self.cache_metadata["total_cache_misses"] += 1
            self._save_cache_metadata()

            return None

        # Загружаем данные из кэша
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Проверяем TTL
            creation_time = datetime.fromisoformat(
                cache_data.get("meta", {}).get("creation_time", "2000-01-01T00:00:00")
            )
            ttl = cache_data.get("meta", {}).get("ttl", self.default_ttl)

            if (datetime.now() - creation_time).total_seconds() > ttl:
                self.logger.info(f"Запись кэша для {analysis_type} устарела, удаляем")

                # Удаляем файл кэша
                os.remove(cache_file)

                # Удаляем запись из метаданных
                if cache_key in self.cache_metadata.get("cache_entries", {}):
                    del self.cache_metadata["cache_entries"][cache_key]

                # Сохраняем обновленные метаданные
                self._save_cache_metadata()

                # Увеличиваем счетчик кэш-миссов
                self.cache_metadata["total_cache_misses"] += 1
                self._save_cache_metadata()

                return None

            # Обновляем метаданные кэша
            if cache_key in self.cache_metadata.get("cache_entries", {}):
                # Увеличиваем счетчик хитов и usefulness
                self.cache_metadata["cache_entries"][cache_key]["hits"] += 1
                self.cache_metadata["cache_entries"][cache_key]["usefulness"] += 1.0

                # Добавляем сэкономленную стоимость
                cost = cache_data.get("meta", {}).get("cost", 0.0)
                self.cache_metadata["total_cost_saved"] += cost
                self.cache_metadata["cache_entries"][cache_key]["cost_saved"] += cost

            # Увеличиваем общий счетчик кэш-хитов
            self.cache_metadata["total_cache_hits"] += 1

            # Сохраняем обновленные метаданные
            self._save_cache_metadata()

            # Добавляем информацию о том, что результат из кэша
            result = cache_data.get("result", {})
            result["from_cache"] = True

            return result

        except Exception as e:
            self.logger.error(f"Ошибка при чтении кэша анализа: {str(e)}")

            # Увеличиваем счетчик кэш-миссов
            self.cache_metadata["total_cache_misses"] += 1
            self._save_cache_metadata()

            return None

    def save_analysis_to_cache(
        self,
        content: str,
        analysis_type: str,
        result: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Сохраняет результат анализа в кэш.

        Args:
            content: Контент, для которого выполнен анализ
            analysis_type: Тип анализа
            result: Результат анализа
            params: Дополнительные параметры анализа (опционально)
            ttl: Время жизни кэша в секундах (опционально)

        Returns:
            bool: True если сохранение успешно, иначе False
        """
        # Создаем ключ кэша
        cache_key = self._create_cache_key(content, analysis_type, params)

        # Путь к файлу кэша
        cache_file = os.path.join(self.analysis_cache_dir, f"{cache_key}.json")

        # Создаем метаданные
        meta = {
            "content_hash": self._hash_content(content),
            "analysis_type": analysis_type,
            "params": params or {},
            "creation_time": datetime.now().isoformat(),
            "ttl": ttl or self.default_ttl,
            "cost": result.get("cost", 0.0),
        }

        # Создаем данные для сохранения
        cache_data = {"meta": meta, "result": result}

        # Сохраняем данные в кэш
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            # Обновляем метаданные кэша
            self.cache_metadata.setdefault("cache_entries", {})[cache_key] = {
                "type": "analysis",
                "analysis_type": analysis_type,
                "creation_time": meta["creation_time"],
                "ttl": meta["ttl"],
                "hits": 0,
                "usefulness": 1.0,  # Начальная полезность
                "cost_saved": 0.0,
            }

            # Сохраняем обновленные метаданные
            self._save_cache_metadata()

            # Проверяем размер кэша
            self._check_cache_size()

            return True

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в кэш анализа: {str(e)}")
            return False

    def get_query_from_cache(
        self, prompt: str, model: str, max_tokens: int, temperature: float
    ) -> Optional[Dict[str, Any]]:
        """
        Получает результат запроса к LLM из кэша.

        Args:
            prompt: Промпт для LLM
            model: Название модели
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации

        Returns:
            Optional[Dict[str, Any]]: Результат запроса из кэша или None, если не найден
        """
        # Создаем ключ кэша
        params = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        cache_key = self._create_cache_key(prompt, "llm_query", params)

        # Путь к файлу кэша
        cache_file = os.path.join(self.query_cache_dir, f"{cache_key}.json")

        # Проверяем наличие файла кэша
        if not os.path.exists(cache_file):
            # Увеличиваем счетчик кэш-миссов
            self.cache_metadata["total_cache_misses"] += 1
            self._save_cache_metadata()

            return None

        # Загружаем данные из кэша
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Проверяем TTL
            creation_time = datetime.fromisoformat(
                cache_data.get("meta", {}).get("creation_time", "2000-01-01T00:00:00")
            )
            ttl = cache_data.get("meta", {}).get("ttl", self.default_ttl)

            if (datetime.now() - creation_time).total_seconds() > ttl:
                self.logger.info(f"Запись кэша для запроса к LLM устарела, удаляем")

                # Удаляем файл кэша
                os.remove(cache_file)

                # Удаляем запись из метаданных
                if cache_key in self.cache_metadata.get("cache_entries", {}):
                    del self.cache_metadata["cache_entries"][cache_key]

                # Сохраняем обновленные метаданные
                self._save_cache_metadata()

                # Увеличиваем счетчик кэш-миссов
                self.cache_metadata["total_cache_misses"] += 1
                self._save_cache_metadata()

                return None

            # Обновляем метаданные кэша
            if cache_key in self.cache_metadata.get("cache_entries", {}):
                # Увеличиваем счетчик хитов и usefulness
                self.cache_metadata["cache_entries"][cache_key]["hits"] += 1
                self.cache_metadata["cache_entries"][cache_key]["usefulness"] += 1.0

                # Добавляем сэкономленную стоимость
                cost = cache_data.get("meta", {}).get("cost", 0.0)
                self.cache_metadata["total_cost_saved"] += cost
                self.cache_metadata["cache_entries"][cache_key]["cost_saved"] += cost

            # Увеличиваем общий счетчик кэш-хитов
            self.cache_metadata["total_cache_hits"] += 1

            # Сохраняем обновленные метаданные
            self._save_cache_metadata()

            # Добавляем информацию о том, что результат из кэша
            result = cache_data.get("result", {})
            result["from_cache"] = True

            return result

        except Exception as e:
            self.logger.error(f"Ошибка при чтении кэша запроса к LLM: {str(e)}")

            # Увеличиваем счетчик кэш-миссов
            self.cache_metadata["total_cache_misses"] += 1
            self._save_cache_metadata()

            return None

    def save_query_to_cache(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        result: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Сохраняет результат запроса к LLM в кэш.

        Args:
            prompt: Промпт для LLM
            model: Название модели
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации
            result: Результат запроса
            ttl: Время жизни кэша в секундах (опционально)

        Returns:
            bool: True если сохранение успешно, иначе False
        """
        # Создаем ключ кэша
        params = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        cache_key = self._create_cache_key(prompt, "llm_query", params)

        # Путь к файлу кэша
        cache_file = os.path.join(self.query_cache_dir, f"{cache_key}.json")

        # Создаем метаданные
        meta = {
            "prompt_hash": self._hash_content(prompt),
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "creation_time": datetime.now().isoformat(),
            "ttl": ttl or self.default_ttl,
            "cost": result.get("cost", 0.0),
        }

        # Создаем данные для сохранения
        cache_data = {"meta": meta, "result": result}

        # Сохраняем данные в кэш
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            # Обновляем метаданные кэша
            self.cache_metadata.setdefault("cache_entries", {})[cache_key] = {
                "type": "query",
                "model": model,
                "creation_time": meta["creation_time"],
                "ttl": meta["ttl"],
                "hits": 0,
                "usefulness": 1.0,  # Начальная полезность
                "cost_saved": 0.0,
            }

            # Сохраняем обновленные метаданные
            self._save_cache_metadata()

            # Проверяем размер кэша
            self._check_cache_size()

            return True

        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в кэш запроса к LLM: {str(e)}")
            return False

    def _create_cache_key(
        self, content: str, type_name: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Создает ключ кэша.

        Args:
            content: Контент для кэширования
            type_name: Тип контента
            params: Дополнительные параметры (опционально)

        Returns:
            str: Ключ кэша
        """
        # Получаем хэш контента
        content_hash = self._hash_content(content)

        # Если есть параметры, получаем их хэш
        params_hash = ""
        if params:
            # Сортируем ключи для стабильного хэширования
            sorted_params = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:8]

        # Формируем ключ кэша
        cache_key = f"{type_name}_{content_hash}"
        if params_hash:
            cache_key += f"_{params_hash}"

        return cache_key

    def _hash_content(self, content: str) -> str:
        """
        Создает хэш контента.

        Args:
            content: Контент для хэширования

        Returns:
            str: Хэш контента
        """
        return hashlib.md5(content.encode()).hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования кэша.

        Returns:
            Dict[str, Any]: Статистика использования кэша
        """
        # Получаем размер кэша
        cache_size = self._get_directory_size(self.cache_dir) / (1024 * 1024)  # в МБ

        # Получаем количество записей в кэше
        analysis_entries = len(
            [
                f
                for f in os.listdir(self.analysis_cache_dir)
                if os.path.isfile(os.path.join(self.analysis_cache_dir, f))
            ]
        )
        query_entries = len(
            [
                f
                for f in os.listdir(self.query_cache_dir)
                if os.path.isfile(os.path.join(self.query_cache_dir, f))
            ]
        )

        # Формируем статистику
        stats = {
            "total_cache_hits": self.cache_metadata.get("total_cache_hits", 0),
            "total_cache_misses": self.cache_metadata.get("total_cache_misses", 0),
            "total_cost_saved": self.cache_metadata.get("total_cost_saved", 0.0),
            "cache_size_mb": cache_size,
            "max_cache_size_mb": self.max_cache_size_mb,
            "analysis_entries": analysis_entries,
            "query_entries": query_entries,
            "total_entries": analysis_entries + query_entries,
            "last_cleanup": self.cache_metadata.get("last_cleanup", ""),
            "cache_hit_rate": self._calculate_hit_rate(),
        }

        return stats

    def _calculate_hit_rate(self) -> float:
        """
        Рассчитывает процент попаданий в кэш.

        Returns:
            float: Процент попаданий в кэш (от 0 до 1)
        """
        hits = self.cache_metadata.get("total_cache_hits", 0)
        misses = self.cache_metadata.get("total_cache_misses", 0)

        total = hits + misses

        if total == 0:
            return 0.0

        return hits / total

    def clear_cache(self, older_than_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Очищает кэш.

        Args:
            older_than_days: Удалить записи старше указанного количества дней (опционально)

        Returns:
            Dict[str, Any]: Результат очистки кэша
        """
        if older_than_days is not None:
            self.logger.info(f"Очистка кэша (записи старше {older_than_days} дней)...")

            # Определяем дату отсечения
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            cutoff_date_str = cutoff_date.isoformat()

            # Счетчики удаленных файлов
            deleted_analysis = 0
            deleted_query = 0

            # Удаляем устаревшие файлы в analysis_cache_dir
            for file_name in os.listdir(self.analysis_cache_dir):
                file_path = os.path.join(self.analysis_cache_dir, file_name)

                if os.path.isfile(file_path):
                    # Получаем метаданные файла
                    cache_key = os.path.splitext(file_name)[0]
                    cache_entry = self.cache_metadata.get("cache_entries", {}).get(cache_key, {})

                    # Получаем время создания
                    creation_time = cache_entry.get(
                        "creation_time",
                        datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                    )

                    # Если файл старше даты отсечения, удаляем его
                    if creation_time < cutoff_date_str:
                        try:
                            os.remove(file_path)

                            # Удаляем запись из метаданных
                            if cache_key in self.cache_metadata.get("cache_entries", {}):
                                del self.cache_metadata["cache_entries"][cache_key]

                            deleted_analysis += 1
                        except Exception as e:
                            self.logger.error(
                                f"Ошибка при удалении файла кэша {file_path}: {str(e)}"
                            )

            # Удаляем устаревшие файлы в query_cache_dir
            for file_name in os.listdir(self.query_cache_dir):
                file_path = os.path.join(self.query_cache_dir, file_name)

                if os.path.isfile(file_path):
                    # Получаем метаданные файла
                    cache_key = os.path.splitext(file_name)[0]
                    cache_entry = self.cache_metadata.get("cache_entries", {}).get(cache_key, {})

                    # Получаем время создания
                    creation_time = cache_entry.get(
                        "creation_time",
                        datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                    )

                    # Если файл старше даты отсечения, удаляем его
                    if creation_time < cutoff_date_str:
                        try:
                            os.remove(file_path)

                            # Удаляем запись из метаданных
                            if cache_key in self.cache_metadata.get("cache_entries", {}):
                                del self.cache_metadata["cache_entries"][cache_key]

                            deleted_query += 1
                        except Exception as e:
                            self.logger.error(
                                f"Ошибка при удалении файла кэша {file_path}: {str(e)}"
                            )

            # Сохраняем обновленные метаданные
            self._save_cache_metadata()

            self.logger.info(
                f"Очистка кэша завершена. Удалено {deleted_analysis} файлов анализа и {deleted_query} файлов запросов."
            )

            return {
                "older_than_days": older_than_days,
                "deleted_analysis": deleted_analysis,
                "deleted_query": deleted_query,
                "total_deleted": deleted_analysis + deleted_query,
            }
        else:
            self.logger.info("Полная очистка кэша...")

            # Удаляем все файлы в директориях кэша
            try:
                # Сохраняем статистику перед очисткой
                analysis_count = len(
                    [
                        f
                        for f in os.listdir(self.analysis_cache_dir)
                        if os.path.isfile(os.path.join(self.analysis_cache_dir, f))
                    ]
                )
                query_count = len(
                    [
                        f
                        for f in os.listdir(self.query_cache_dir)
                        if os.path.isfile(os.path.join(self.query_cache_dir, f))
                    ]
                )

                # Очищаем директории кэша
                for file_name in os.listdir(self.analysis_cache_dir):
                    file_path = os.path.join(self.analysis_cache_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                for file_name in os.listdir(self.query_cache_dir):
                    file_path = os.path.join(self.query_cache_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                # Сбрасываем метаданные кэша
                self.cache_metadata = {
                    "last_cleanup": datetime.now().isoformat(),
                    "total_cache_hits": 0,
                    "total_cache_misses": 0,
                    "total_cost_saved": 0.0,
                    "cache_entries": {},
                }

                # Сохраняем обновленные метаданные
                self._save_cache_metadata()

                self.logger.info(
                    f"Полная очистка кэша завершена. Удалено {analysis_count} файлов анализа и {query_count} файлов запросов."
                )

                return {
                    "full_cleanup": True,
                    "deleted_analysis": analysis_count,
                    "deleted_query": query_count,
                    "total_deleted": analysis_count + query_count,
                }
            except Exception as e:
                self.logger.error(f"Ошибка при полной очистке кэша: {str(e)}")

                return {
                    "full_cleanup": True,
                    "error": str(e),
                    "deleted_analysis": 0,
                    "deleted_query": 0,
                    "total_deleted": 0,
                }
