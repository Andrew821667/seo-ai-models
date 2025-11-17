"""
Система создания эффективных промптов для различных LLM.

Модуль предоставляет функционал для генерации оптимизированных промптов
для различных задач и LLM-провайдеров.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from string import Template

# Импортируем компоненты из common
from ..common.constants import DEFAULT_PROMPTS, LLM_MODELS
from ..common.exceptions import PromptGenerationError


class PromptTemplate:
    """
    Шаблон промпта с возможностью форматирования.
    """

    def __init__(self, template: str, name: str = None):
        """
        Инициализирует шаблон промпта.

        Args:
            template: Текст шаблона
            name: Название шаблона (опционально)
        """
        self.template = template
        self.name = name

    def format(self, **kwargs) -> str:
        """
        Форматирует шаблон с переданными параметрами.

        Args:
            **kwargs: Параметры для форматирования

        Returns:
            str: Отформатированный промпт

        Raises:
            PromptGenerationError: При ошибке форматирования
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise PromptGenerationError(self.name, f"Отсутствует обязательный параметр: {e}")
        except Exception as e:
            raise PromptGenerationError(self.name, f"Ошибка форматирования: {e}")


class PromptGenerator:
    """
    Генератор промптов для различных задач и LLM-провайдеров.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Инициализирует генератор промптов.

        Args:
            templates_dir: Путь к директории с шаблонами промптов (опционально)
        """
        # Загружаем дефолтные промпты
        self.templates = {}

        for analysis_type, template_text in DEFAULT_PROMPTS.items():
            self.templates[analysis_type] = PromptTemplate(template_text, name=analysis_type)

        # Если указана директория с шаблонами, загружаем их
        if templates_dir and os.path.exists(templates_dir):
            self.load_templates_from_directory(templates_dir)

    def load_templates_from_directory(self, templates_dir: str) -> None:
        """
        Загружает шаблоны промптов из директории.

        Args:
            templates_dir: Путь к директории с шаблонами
        """
        for filename in os.listdir(templates_dir):
            if filename.endswith(".json"):
                template_path = os.path.join(templates_dir, filename)
                try:
                    with open(template_path, "r", encoding="utf-8") as f:
                        template_data = json.load(f)

                    # Проверяем наличие обязательных полей
                    if "name" in template_data and "template" in template_data:
                        self.templates[template_data["name"]] = PromptTemplate(
                            template_data["template"], name=template_data["name"]
                        )
                except Exception as e:
                    print(f"Ошибка загрузки шаблона {filename}: {e}")

    def add_template(self, name: str, template: str) -> None:
        """
        Добавляет новый шаблон промпта.

        Args:
            name: Название шаблона
            template: Текст шаблона
        """
        self.templates[name] = PromptTemplate(template, name=name)

    def remove_template(self, name: str) -> None:
        """
        Удаляет шаблон промпта.

        Args:
            name: Название шаблона

        Raises:
            KeyError: Если шаблон с указанным названием не существует
        """
        if name in self.templates:
            del self.templates[name]
        else:
            raise KeyError(f"Шаблон с названием '{name}' не существует")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Возвращает шаблон промпта.

        Args:
            name: Название шаблона

        Returns:
            Optional[PromptTemplate]: Шаблон промпта или None, если шаблон не найден
        """
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """
        Возвращает список доступных шаблонов.

        Returns:
            List[str]: Список названий шаблонов
        """
        return list(self.templates.keys())

    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """
        Генерирует промпт на основе шаблона.

        Args:
            template_name: Название шаблона
            **kwargs: Параметры для форматирования

        Returns:
            str: Отформатированный промпт

        Raises:
            KeyError: Если шаблон с указанным названием не существует
            PromptGenerationError: При ошибке форматирования
        """
        if template_name not in self.templates:
            raise KeyError(f"Шаблон с названием '{template_name}' не существует")

        return self.templates[template_name].format(**kwargs)

    def optimize_for_provider(self, prompt: str, provider: str, model: str) -> str:
        """
        Оптимизирует промпт для конкретного провайдера и модели.

        Args:
            prompt: Исходный промпт
            provider: Название провайдера
            model: Название модели

        Returns:
            str: Оптимизированный промпт
        """
        # Применяем специфические оптимизации в зависимости от провайдера
        if provider == "openai":
            # Для OpenAI не требуется особых изменений
            return prompt

        elif provider == "anthropic":
            # Для Claude можно добавить специфические инструкции
            anthropic_prefix = (
                "Пожалуйста, ответь на следующий вопрос максимально подробно "
                "и с учетом всех деталей. "
            )
            return anthropic_prefix + prompt

        elif provider == "gigachat":
            # Для GigaChat адаптируем промпт
            return prompt

        elif provider == "local":
            # Для локальных моделей может потребоваться упрощение
            if "llama" in model:
                # Для Llama моделей
                return prompt
            elif "deepseek" in model:
                # Для DeepSeek моделей
                return prompt
            else:
                # Для остальных локальных моделей
                return prompt

        # По умолчанию возвращаем исходный промпт
        return prompt

    def generate_analysis_prompt(
        self, analysis_type: str, content: str, provider: str = None, model: str = None
    ) -> str:
        """
        Генерирует промпт для анализа контента.

        Args:
            analysis_type: Тип анализа (compatibility, citability, structure, eeat, semantic)
            content: Содержимое для анализа
            provider: Название провайдера (опционально)
            model: Название модели (опционально)

        Returns:
            str: Отформатированный промпт

        Raises:
            KeyError: Если шаблон для указанного типа анализа не существует
            PromptGenerationError: При ошибке форматирования
        """
        # Генерируем промпт для анализа
        prompt = self.generate_prompt(analysis_type, content=content)

        # Если указаны провайдер и модель, оптимизируем промпт
        if provider and model:
            prompt = self.optimize_for_provider(prompt, provider, model)

        return prompt
