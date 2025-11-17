"""
Усовершенствованный семантический анализатор с использованием NLP моделей.
Обеспечивает глубокий анализ контента, извлечение сущностей и определение тематики.
"""

import logging
import time
import re
import string
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from collections import Counter
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnhancedSemanticAnalyzer:
    """
    Усовершенствованный семантический анализатор, использующий NLP подходы
    для глубокого анализа контента.
    """

    def __init__(
        self,
        language: str = "en",
        use_spacy: bool = True,
        use_nltk: bool = True,
        use_gensim: bool = False,
        spacy_model: str = "en_core_web_sm",
        enable_entity_recognition: bool = True,
        enable_sentiment_analysis: bool = True,
        custom_stopwords: Optional[List[str]] = None,
    ):
        """
        Инициализация семантического анализатора.

        Args:
            language: Язык анализа ('en', 'ru', и т.д.)
            use_spacy: Использовать ли SpaCy для NLP
            use_nltk: Использовать ли NLTK для NLP
            use_gensim: Использовать ли Gensim для тематического моделирования
            spacy_model: Модель SpaCy для загрузки
            enable_entity_recognition: Включить распознавание именованных сущностей
            enable_sentiment_analysis: Включить анализ тональности
            custom_stopwords: Пользовательский список стоп-слов
        """
        self.language = language
        self.use_spacy = use_spacy
        self.use_nltk = use_nltk
        self.use_gensim = use_gensim
        self.spacy_model = spacy_model
        self.enable_entity_recognition = enable_entity_recognition
        self.enable_sentiment_analysis = enable_sentiment_analysis

        # Инициализация NLP компонентов
        self.nlp = None
        self.stopwords = set()
        self.custom_stopwords = custom_stopwords or []

        # Загружаем необходимые компоненты
        self._load_nlp_components()

        logger.info(f"EnhancedSemanticAnalyzer initialized for language: {language}")
        logger.info(f"NLP components: SpaCy: {use_spacy}, NLTK: {use_nltk}, Gensim: {use_gensim}")

    def _load_nlp_components(self):
        """
        Загружает необходимые NLP компоненты.
        """
        # Загружаем стоп-слова
        try:
            # Пытаемся загрузить стоп-слова из NLTK
            if self.use_nltk:
                try:
                    import nltk
                    from nltk.corpus import stopwords

                    try:
                        nltk.data.find(f"corpora/stopwords")
                    except LookupError:
                        nltk.download("stopwords")

                    self.stopwords.update(stopwords.words(self._map_language_to_nltk()))
                    logger.info(f"Loaded {len(self.stopwords)} stopwords from NLTK")
                except Exception as e:
                    logger.warning(f"Error loading NLTK stopwords: {str(e)}")

            # Загружаем SpaCy
            if self.use_spacy:
                try:
                    import spacy

                    # В реальной системе здесь была бы загрузка модели
                    # self.nlp = spacy.load(self.spacy_model)
                    logger.info(f"SpaCy model would be loaded here: {self.spacy_model}")

                    # Добавляем стоп-слова из SpaCy
                    # self.stopwords.update(self.nlp.Defaults.stop_words)
                except Exception as e:
                    logger.warning(f"Error loading SpaCy model: {str(e)}")
                    self.use_spacy = False

            # Добавляем пользовательские стоп-слова
            if self.custom_stopwords:
                self.stopwords.update(self.custom_stopwords)

        except Exception as e:
            logger.error(f"Error initializing NLP components: {str(e)}")

    def _map_language_to_nltk(self) -> str:
        """
        Преобразует код языка в формат, понятный NLTK.

        Returns:
            str: Код языка для NLTK
        """
        language_map = {
            "en": "english",
            "ru": "russian",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "nl": "dutch",
            "sv": "swedish",
            "fi": "finnish",
            "da": "danish",
            "hu": "hungarian",
            "no": "norwegian",
        }

        return language_map.get(self.language.lower(), "english")

    def analyze_text(self, text: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Выполняет комплексный семантический анализ текста.

        Args:
            text: Текст для анализа
            keywords: Список целевых ключевых слов

        Returns:
            Dict[str, Any]: Результаты анализа
        """
        if not text:
            return {
                "error": "Empty text",
                "semantic_density": 0,
                "semantic_coverage": 0,
                "topical_coherence": 0,
                "contextual_relevance": 0,
            }

        start_time = time.time()

        # Предварительная обработка текста
        clean_text = self._preprocess_text(text)

        # Токенизация и лемматизация
        tokens, lemmas, pos_tags = self._tokenize_and_lemmatize(clean_text)

        # Извлечение ключевых слов
        extracted_keywords = self._extract_keywords(lemmas, tokens, pos_tags)

        # Определение тематики
        topics = self._identify_topics(lemmas, extracted_keywords)

        # Анализ связности и согласованности
        coherence = self._analyze_coherence(lemmas, tokens, topics)

        # Анализ сложности текста
        complexity = self._analyze_complexity(tokens, lemmas, pos_tags)

        # Именованные сущности
        entities = self._extract_entities(clean_text) if self.enable_entity_recognition else {}

        # Сентимент-анализ
        sentiment = (
            self._analyze_sentiment(clean_text)
            if self.enable_sentiment_analysis
            else {"neutral": 1.0}
        )

        # Расчет метрик относительно целевых ключевых слов
        target_keywords_metrics = (
            self._analyze_target_keywords(clean_text, tokens, lemmas, keywords) if keywords else {}
        )

        # Рассчитываем общие семантические метрики
        semantic_density = self._calculate_semantic_density(extracted_keywords, tokens)
        semantic_coverage = self._calculate_semantic_coverage(
            extracted_keywords, topics, target_keywords_metrics
        )
        topical_coherence = self._calculate_topical_coherence(coherence, topics)
        contextual_relevance = self._calculate_contextual_relevance(
            target_keywords_metrics, coherence
        )

        # Формируем результат
        result = {
            "semantic_density": semantic_density,
            "semantic_coverage": semantic_coverage,
            "topical_coherence": topical_coherence,
            "contextual_relevance": contextual_relevance,
            "keywords": extracted_keywords,
            "topics": topics,
            "coherence": coherence,
            "complexity": complexity,
            "entities": entities,
            "sentiment": sentiment,
            "target_keywords_metrics": target_keywords_metrics,
            "processing_time": time.time() - start_time,
        }

        return result

    def generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Генерирует рекомендации на основе результатов семантического анализа.

        Args:
            analysis_result: Результаты семантического анализа

        Returns:
            List[str]: Список рекомендаций
        """
        recommendations = []

        # Проверяем семантическую плотность
        semantic_density = analysis_result.get("semantic_density", 0)
        if semantic_density < 0.3:
            recommendations.append(
                "Увеличьте семантическую плотность, добавив больше ключевых слов и тематических терминов"
            )
        elif semantic_density > 0.7:
            recommendations.append(
                "Семантическая плотность слишком высока, сделайте текст более естественным"
            )

        # Проверяем семантический охват
        semantic_coverage = analysis_result.get("semantic_coverage", 0)
        if semantic_coverage < 0.4:
            recommendations.append(
                "Расширьте семантическое поле, включив больше связанных терминов и понятий"
            )

        # Проверяем тематическую когерентность
        topical_coherence = analysis_result.get("topical_coherence", 0)
        if topical_coherence < 0.5:
            recommendations.append(
                "Улучшите связность текста, обеспечив более логичные переходы между темами"
            )

        # Проверяем контекстуальную релевантность
        contextual_relevance = analysis_result.get("contextual_relevance", 0)
        if contextual_relevance < 0.5:
            recommendations.append(
                "Усильте контекстуальную релевантность, включив больше терминов, связанных с целевыми ключевыми словами"
            )

        # Проверяем сложность текста
        complexity = analysis_result.get("complexity", {})
        readability = complexity.get("readability", 0.5)
        if readability < 0.3:
            recommendations.append(
                "Упростите текст, используя более короткие предложения и общеупотребительные слова"
            )
        elif readability > 0.8 and complexity.get("sentence_length_avg", 0) < 10:
            recommendations.append(
                "Текст может быть слишком простым, добавьте больше содержательных деталей"
            )

        # Проверяем тональность
        sentiment = analysis_result.get("sentiment", {})
        if sentiment.get("negative", 0) > 0.6:
            recommendations.append(
                "Текст имеет преимущественно негативную тональность, рассмотрите возможность более позитивной подачи"
            )

        # Проверяем метрики целевых ключевых слов
        target_keywords_metrics = analysis_result.get("target_keywords_metrics", {})
        if target_keywords_metrics.get("average_prominence", 0) < 0.3:
            recommendations.append(
                "Улучшите расположение ключевых слов, включив их в заголовки и начало абзацев"
            )

        if target_keywords_metrics.get("average_density", 0) < 0.5:
            recommendations.append(
                "Увеличьте плотность ключевых слов, особенно основных целевых запросов"
            )

        # Если рекомендаций мало, добавляем общие советы
        if len(recommendations) < 2:
            recommendations.append(
                "Добавьте больше тематически связанных терминов для улучшения семантического охвата"
            )
            recommendations.append(
                "Включите разнообразные формы ключевых слов (синонимы, падежи, времена)"
            )

        return recommendations

    def analyze_keyword_relevance(self, text: str, keyword: str) -> Dict[str, float]:
        """
        Анализирует релевантность ключевого слова в тексте.

        Args:
            text: Анализируемый текст
            keyword: Ключевое слово

        Returns:
            Dict[str, float]: Метрики релевантности
        """
        # Предварительная обработка
        clean_text = self._preprocess_text(text)
        keyword_clean = self._preprocess_text(keyword)

        # Токенизация и лемматизация
        tokens, lemmas, _ = self._tokenize_and_lemmatize(clean_text)
        keyword_tokens = self._simple_tokenize(keyword_clean)

        # Поиск точных и частичных совпадений
        exact_matches = 0
        partial_matches = 0

        # Для многословных ключевых слов
        if len(keyword_tokens) > 1:
            keyword_pattern = r"" + r"\s+".join([re.escape(t) for t in keyword_tokens]) + r""
            exact_matches = len(re.findall(keyword_pattern, clean_text, re.IGNORECASE))

            # Для частичных совпадений проверяем наличие отдельных токенов
            for token in keyword_tokens:
                if token in lemmas:
                    partial_matches += 1

            partial_matches = (
                partial_matches / len(keyword_tokens) if len(keyword_tokens) > 0 else 0
            )
        else:
            # Для однословных ключевых слов
            keyword_token = keyword_tokens[0] if keyword_tokens else ""
            keyword_pattern = r"" + re.escape(keyword_token) + r""
            exact_matches = len(re.findall(keyword_pattern, clean_text, re.IGNORECASE))

            # Проверяем лемматизированные формы
            if keyword_token in lemmas:
                partial_matches = 1.0

        # Расчет плотности ключевых слов
        word_count = len(tokens)
        density = exact_matches / word_count if word_count > 0 else 0

        # Проверка наличия ключевого слова в разных частях текста
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs_with_keyword = sum(
            1 for p in paragraphs if re.search(keyword_pattern, p, re.IGNORECASE)
        )
        paragraphs_with_keyword = sum(
            1 for p in paragraphs if re.search(keyword_pattern, p, re.IGNORECASE)
        )
        paragraph_coverage = paragraphs_with_keyword / len(paragraphs) if paragraphs else 0

        # Проверка наличия ключевого слова в заголовках
        headings = re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", text, re.IGNORECASE) or []
        headings_with_keyword = sum(
            1 for h in headings if re.search(keyword_pattern, h, re.IGNORECASE)
        )
        heading_presence = headings_with_keyword / len(headings) if headings else 0

        # Расчет общей релевантности
        relevance = (
            (density * 0.3 + paragraph_coverage * 0.3 + heading_presence * 0.4)
            if word_count > 0
            else 0
        )

        return {
            "exact_matches": exact_matches,
            "density": density,
            "partial_matches": partial_matches,
            "paragraph_coverage": paragraph_coverage,
            "heading_presence": heading_presence,
            "relevance": relevance,
        }

    def compare_texts_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Сравнивает два текста для определения их семантического сходства.

        Args:
            text1: Первый текст
            text2: Второй текст

        Returns:
            Dict[str, float]: Метрики сходства
        """
        # Предварительная обработка текстов
        clean_text1 = self._preprocess_text(text1)
        clean_text2 = self._preprocess_text(text2)

        # Токенизация и лемматизация
        tokens1, lemmas1, _ = self._tokenize_and_lemmatize(clean_text1)
        tokens2, lemmas2, _ = self._tokenize_and_lemmatize(clean_text2)

        # Создаем словари частот
        lemmas1_count = Counter(lemmas1)
        lemmas2_count = Counter(lemmas2)

        # Множества уникальных лемм
        unique_lemmas1 = set(lemmas1_count.keys())
        unique_lemmas2 = set(lemmas2_count.keys())

        # Общие леммы
        common_lemmas = unique_lemmas1.intersection(unique_lemmas2)

        # Мера Жаккара
        jaccard = (
            len(common_lemmas) / (len(unique_lemmas1) + len(unique_lemmas2) - len(common_lemmas))
            if (len(unique_lemmas1) + len(unique_lemmas2) - len(common_lemmas)) > 0
            else 0
        )

        # Косинусное сходство
        cosine = self._calculate_cosine_similarity(lemmas1_count, lemmas2_count)

        # Процентное пересечение контента
        content_overlap = self._calculate_content_overlap(tokens1, tokens2)

        # Тематическое сходство
        topic_similarity = self._calculate_topic_similarity(lemmas1, lemmas2)

        # Общее семантическое сходство (взвешенное среднее)
        weighted_similarity = (
            jaccard * 0.3 + cosine * 0.4 + content_overlap * 0.2 + topic_similarity * 0.1
        )

        return {
            "jaccard_similarity": jaccard,
            "cosine_similarity": cosine,
            "content_overlap": content_overlap,
            "topic_similarity": topic_similarity,
            "weighted_similarity": weighted_similarity,
        }

    def _preprocess_text(self, text: str) -> str:
        """
        Предварительная обработка текста.

        Args:
            text: Исходный текст

        Returns:
            str: Очищенный текст
        """
        if not text:
            return ""

        # Преобразуем HTML-сущности
        text = re.sub(r"&[a-z]+;", " ", text)

        # Удаляем HTML-теги
        text = re.sub(r"<[^>]+>", " ", text)

        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем URL
        text = re.sub(r"https?://\S+", " ", text)

        # Удаляем email
        text = re.sub(r"\S+@\S+", " ", text)

        # Удаляем специальные символы и цифры (сохраняем буквы и пробелы)
        text = re.sub(r"[^\w\s]", " ", text)

        # Удаляем множественные пробелы
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _tokenize_and_lemmatize(self, text: str) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Токенизация и лемматизация текста.

        Args:
            text: Текст для обработки

        Returns:
            Tuple[List[str], List[str], Dict[str, str]]: Кортеж (токены, леммы, POS-теги)
        """
        if not text:
            return [], [], {}

        # Простая токенизация в качестве запасного варианта
        tokens = self._simple_tokenize(text)

        # Базовая лемматизация (учитывая отсутствие SpaCy)
        # В реальной системе здесь была бы настоящая лемматизация
        lemmas = tokens.copy()

        # POS-теги (часть речи)
        pos_tags = {token: "NOUN" for token in tokens}  # Упрощение для демонстрации

        return tokens, lemmas, pos_tags

    def _simple_tokenize(self, text: str) -> List[str]:
        """
        Простая токенизация текста.

        Args:
            text: Текст для токенизации

        Returns:
            List[str]: Список токенов
        """
        if not text:
            return []

        # Разбиваем по пробелам
        tokens = text.split()

        # Фильтруем пустые токены и стоп-слова
        tokens = [t for t in tokens if t and t not in self.stopwords and len(t) > 1]

        return tokens

    def _extract_keywords(
        self, lemmas: List[str], tokens: List[str], pos_tags: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Извлечение ключевых слов с весами.

        Args:
            lemmas: Лемматизированные токены
            tokens: Исходные токены
            pos_tags: POS-теги

        Returns:
            Dict[str, float]: Ключевые слова с весами
        """
        if not lemmas:
            return {}

        # Подсчитываем частоту слов
        word_freq = Counter(lemmas)

        # Общее количество слов
        total_words = len(lemmas)

        # Рассчитываем веса (нормализованные)
        keywords = {}
        max_freq = word_freq.most_common(1)[0][1] if word_freq else 1

        for word, freq in word_freq.most_common(30):  # Берем топ-30 слов
            # Нормализуем вес от 0 до 1
            weight = freq / max_freq
            keywords[word] = weight

        return keywords

    def _identify_topics(
        self, lemmas: List[str], keywords: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Определение основных тем текста.

        Args:
            lemmas: Лемматизированные токены
            keywords: Извлеченные ключевые слова с весами

        Returns:
            List[Dict[str, Any]]: Список тем с ключевыми словами
        """
        if not lemmas or not keywords:
            return []

        # Группируем ключевые слова по предполагаемым темам
        # В реальной системе здесь было бы тематическое моделирование (LDA, TF-IDF и т.д.)

        # Упрощенная демонстрационная группировка тем
        topics = []

        # Берем топ ключевые слова для первой темы
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        topic1 = {"name": "Main Topic", "keywords": {k: v for k, v in top_keywords}, "weight": 1.0}
        topics.append(topic1)

        return topics

    def _analyze_coherence(
        self, lemmas: List[str], tokens: List[str], topics: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Анализ связности и согласованности текста.

        Args:
            lemmas: Лемматизированные токены
            tokens: Исходные токены
            topics: Определенные темы

        Returns:
            Dict[str, float]: Метрики связности
        """
        if not lemmas or not tokens:
            return {"overall_coherence": 0, "topic_consistency": 0, "flow_score": 0}

        # Упрощенные метрики связности для демонстрации

        # Оценка тематической последовательности
        topic_consistency = 0.7  # Средняя оценка по умолчанию

        # Оценка логического потока
        flow_score = 0.65  # Средняя оценка по умолчанию

        # Общая оценка связности
        overall_coherence = (topic_consistency + flow_score) / 2

        return {
            "overall_coherence": overall_coherence,
            "topic_consistency": topic_consistency,
            "flow_score": flow_score,
        }

    def _analyze_complexity(
        self, tokens: List[str], lemmas: List[str], pos_tags: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Анализ сложности текста.

        Args:
            tokens: Исходные токены
            lemmas: Лемматизированные токены
            pos_tags: POS-теги

        Returns:
            Dict[str, Any]: Метрики сложности
        """
        if not tokens:
            return {"readability": 0.5, "lexical_diversity": 0, "sentence_length_avg": 0}

        # Лексическое разнообразие (соотношение уникальных слов к общему количеству)
        unique_lemmas = set(lemmas)
        lexical_diversity = len(unique_lemmas) / len(lemmas) if lemmas else 0

        # Средняя длина предложения (упрощенно)
        # В реальной системе здесь был бы более сложный алгоритм
        sentence_length_avg = 15  # Среднее значение по умолчанию

        # Оценка читабельности (упрощенно)
        # В реальной системе здесь была бы формула оценки читабельности
        readability = 0.75  # Средняя оценка по умолчанию

        return {
            "readability": readability,
            "lexical_diversity": lexical_diversity,
            "sentence_length_avg": sentence_length_avg,
        }

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Извлечение именованных сущностей.

        Args:
            text: Текст для анализа

        Returns:
            Dict[str, List[str]]: Именованные сущности по категориям
        """
        # В реальной системе здесь было бы использование NER от SpaCy или другой библиотеки
        # Возвращаем заглушку для демонстрации
        return {"PERSON": [], "ORG": [], "LOC": [], "DATE": [], "PRODUCT": []}

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Анализ тональности текста.

        Args:
            text: Текст для анализа

        Returns:
            Dict[str, float]: Оценки тональности
        """
        # В реальной системе здесь был бы полноценный сентимент-анализ
        # Возвращаем заглушку для демонстрации
        return {"positive": 0.6, "negative": 0.1, "neutral": 0.3}

    def _analyze_target_keywords(
        self, text: str, tokens: List[str], lemmas: List[str], keywords: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Анализ целевых ключевых слов.

        Args:
            text: Исходный текст
            tokens: Токены
            lemmas: Леммы
            keywords: Целевые ключевые слова

        Returns:
            Dict[str, Any]: Метрики ключевых слов
        """
        if not keywords or not text:
            return {}

        # Метрики для каждого ключевого слова
        keyword_metrics = {}

        for keyword in keywords:
            keyword_metrics[keyword] = self.analyze_keyword_relevance(text, keyword)

        # Средние метрики
        avg_density = (
            sum(m["density"] for m in keyword_metrics.values()) / len(keyword_metrics)
            if keyword_metrics
            else 0
        )
        avg_relevance = (
            sum(m["relevance"] for m in keyword_metrics.values()) / len(keyword_metrics)
            if keyword_metrics
            else 0
        )
        avg_prominence = (
            sum(m["heading_presence"] for m in keyword_metrics.values()) / len(keyword_metrics)
            if keyword_metrics
            else 0
        )

        return {
            "keywords": keyword_metrics,
            "average_density": avg_density,
            "average_relevance": avg_relevance,
            "average_prominence": avg_prominence,
        }

    def _calculate_semantic_density(self, keywords: Dict[str, float], tokens: List[str]) -> float:
        """
        Расчет семантической плотности.

        Args:
            keywords: Извлеченные ключевые слова с весами
            tokens: Токены текста

        Returns:
            float: Оценка семантической плотности
        """
        if not keywords or not tokens:
            return 0

        # Сумма весов ключевых слов, нормализованная по длине текста
        keywords_weight_sum = sum(keywords.values())

        # Нормализация по количеству ключевых слов
        normalized_weight = keywords_weight_sum / len(keywords) if keywords else 0

        # Доля ключевых слов в тексте
        keyword_presence = len(keywords) / len(tokens) if tokens else 0

        # Итоговая оценка
        semantic_density = normalized_weight * 0.6 + keyword_presence * 0.4

        return min(1.0, semantic_density)

    def _calculate_semantic_coverage(
        self,
        keywords: Dict[str, float],
        topics: List[Dict[str, Any]],
        target_keywords_metrics: Dict[str, Any],
    ) -> float:
        """
        Расчет семантического охвата.

        Args:
            keywords: Извлеченные ключевые слова с весами
            topics: Определенные темы
            target_keywords_metrics: Метрики целевых ключевых слов

        Returns:
            float: Оценка семантического охвата
        """
        if not keywords:
            return 0

        # Количество извлеченных ключевых слов
        keyword_count = len(keywords)

        # Разнообразие тем
        topic_coverage = len(topics) / 3 if topics else 0  # Нормализуем по 3 возможным темам

        # Оценка на основе целевых ключевых слов
        target_coverage = target_keywords_metrics.get("average_relevance", 0)

        # Итоговая оценка
        semantic_coverage = (
            (keyword_count / 20) * 0.4 + topic_coverage * 0.3 + target_coverage * 0.3
        )

        return min(1.0, semantic_coverage)

    def _calculate_topical_coherence(
        self, coherence: Dict[str, float], topics: List[Dict[str, Any]]
    ) -> float:
        """
        Расчет тематической когерентности.

        Args:
            coherence: Метрики связности
            topics: Определенные темы

        Returns:
            float: Оценка тематической когерентности
        """
        if not coherence or not topics:
            return 0

        # Используем метрики связности
        topic_consistency = coherence.get("topic_consistency", 0)
        flow_score = coherence.get("flow_score", 0)

        # Итоговая оценка
        topical_coherence = topic_consistency * 0.7 + flow_score * 0.3

        return topical_coherence

    def _calculate_contextual_relevance(
        self, target_keywords_metrics: Dict[str, Any], coherence: Dict[str, float]
    ) -> float:
        """
        Расчет контекстуальной релевантности.

        Args:
            target_keywords_metrics: Метрики целевых ключевых слов
            coherence: Метрики связности

        Returns:
            float: Оценка контекстуальной релевантности
        """
        if not target_keywords_metrics or not coherence:
            return 0

        # Средняя релевантность ключевых слов
        avg_relevance = target_keywords_metrics.get("average_relevance", 0)

        # Среднее распределение ключевых слов
        avg_prominence = target_keywords_metrics.get("average_prominence", 0)

        # Связность текста
        overall_coherence = coherence.get("overall_coherence", 0)

        # Итоговая оценка
        contextual_relevance = avg_relevance * 0.4 + avg_prominence * 0.3 + overall_coherence * 0.3

        return contextual_relevance

    def _calculate_cosine_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        """
        Расчет косинусного сходства между векторами.

        Args:
            vec1: Первый вектор
            vec2: Второй вектор

        Returns:
            float: Косинусное сходство
        """
        if not vec1 or not vec2:
            return 0

        # Находим общие ключи
        common_keys = set(vec1.keys()).intersection(set(vec2.keys()))

        # Считаем скалярное произведение
        dot_product = sum(vec1[key] * vec2[key] for key in common_keys)

        # Считаем нормы векторов
        norm1 = math.sqrt(sum(val * val for val in vec1.values()))
        norm2 = math.sqrt(sum(val * val for val in vec2.values()))

        # Косинусное сходство
        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def _calculate_content_overlap(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Расчет пересечения контента.

        Args:
            tokens1: Токены первого текста
            tokens2: Токены второго текста

        Returns:
            float: Оценка пересечения контента
        """
        if not tokens1 or not tokens2:
            return 0

        # Множества уникальных токенов
        unique_tokens1 = set(tokens1)
        unique_tokens2 = set(tokens2)

        # Общие токены
        common_tokens = unique_tokens1.intersection(unique_tokens2)

        # Пересечение контента (от 0 до 1)
        smaller_set_size = min(len(unique_tokens1), len(unique_tokens2))
        if smaller_set_size == 0:
            return 0

        return len(common_tokens) / smaller_set_size

    def _calculate_topic_similarity(self, lemmas1: List[str], lemmas2: List[str]) -> float:
        """
        Расчет тематического сходства.

        Args:
            lemmas1: Леммы первого текста
            lemmas2: Леммы второго текста

        Returns:
            float: Оценка тематического сходства
        """
        if not lemmas1 or not lemmas2:
            return 0

        # В реальной системе здесь был бы более сложный алгоритм
        # Для демонстрации используем упрощенный подход через косинусное сходство

        # Создаем словари частот
        lemmas1_count = Counter(lemmas1)
        lemmas2_count = Counter(lemmas2)

        return self._calculate_cosine_similarity(lemmas1_count, lemmas2_count)
