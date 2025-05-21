
"""
Финальное исправление несогласованности оценок читабельности.
"""

from seo_ai_models.models.seo_advisor.analyzers.enhanced_content_analyzer import EnhancedContentAnalyzer

# Полностью переопределяем метод analyze_content
def fixed_analyze_content(self, content, html_content=None):
    """
    Комплексный анализ контента с учетом HTML-структуры и согласованными шкалами читабельности.
    
    Args:
        content: Текстовое содержимое для анализа
        html_content: HTML-версия контента (если доступна)
        
    Returns:
        Dict[str, Any]: Метрики контента
    """
    metrics = {}
    
    # Определяем язык контента
    language = self.text_processor.detect_language(content)
    
    # Если предоставлен HTML, используем его для более глубокого анализа
    if html_content:
        html_metrics = self.text_processor.process_html_content(html_content)
        
        # Объединяем результаты, предпочитая HTML-метрики
        metrics.update({
            'word_count': html_metrics.get('word_count', 0),
            'sentence_count': html_metrics.get('sentence_count', 0),
            'avg_sentence_length': html_metrics.get('avg_sentence_length', 0),
            'headings_count': len(html_metrics.get('headings', [])),
            'images_count': len(html_metrics.get('images', [])),
            'links_count': len(html_metrics.get('links', [])),
            'lists_count': len(html_metrics.get('lists', [])),
            'tables_count': len(html_metrics.get('tables', [])),
            'has_images': len(html_metrics.get('images', [])) > 0,
            'has_lists': len(html_metrics.get('lists', [])) > 0,
            'has_tables': len(html_metrics.get('tables', [])) > 0
        })
        
        # Если в HTML есть больше текста, используем его
        if html_metrics.get('word_count', 0) > len(self.text_processor.tokenize(content)):
            content = html_metrics.get('text', content)
    else:
        # Базовые метрики из текста
        words = self.text_processor.tokenize(content)
        sentences = self.text_processor.split_sentences(content)
        
        metrics['word_count'] = len(words)
        metrics['sentence_count'] = len(sentences)
        metrics['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
    
    # Оценка читабельности с учетом языка
    readability_metrics = self.text_processor.calculate_enhanced_readability(content, language)
    
    # Используем прямое значение из TextProcessor для согласованности
    metrics['readability'] = readability_metrics.get('readability_score', 0)
    metrics['flesch_reading_ease'] = readability_metrics.get('flesch_reading_ease', 0)
    metrics['complexity_level'] = readability_metrics.get('complexity_level', 0)
    
    # Анализ заголовков
    headers = self.text_processor.extract_headers(content)
    metrics['header_score'] = self._calculate_header_score(headers)
    metrics['headers_count'] = len(headers)
    
    # Структурный анализ
    structure = self.text_processor.analyze_text_structure(content)
    metrics.update({
        'structure_score': self._calculate_structure_score(structure),
        'has_introduction': structure.get('has_introduction', False),
        'has_conclusion': structure.get('has_conclusion', False),
        'paragraphs_count': structure.get('paragraphs_count', 0)
    })
    
    # Выявление основных тем
    main_topics = self.text_processor.extract_main_topics(content)
    metrics['main_topics'] = main_topics
    
    # Оценки для SEO и юзабилити
    metrics.update({
        'meta_score': 0.7,  # Заглушка, обычно берется из HTML-метаданных
        'multimedia_score': 0.7 if metrics.get('has_images', False) else 0.3,
        'linking_score': 0.6,  # Заглушка, зависит от анализа ссылок
        
        # Семантический анализ
        'semantic_depth': 0.5,
        'topic_relevance': 0.8,
        'engagement_potential': 0.65
    })
    
    # Проверка согласованности метрик
    consistent_metrics = self.consistency_checker.check_and_fix(metrics)
    
    # Бонусные метрики для улучшенного анализа
    consistent_metrics.update({
        'language': language,
        'content_type': 'html' if html_content else 'text',
        'complexity_score': self._calculate_complexity_score(consistent_metrics),
        'content_density': self._calculate_content_density(consistent_metrics)
    })
    
    return consistent_metrics

# Заменяем метод
EnhancedContentAnalyzer.analyze_content = fixed_analyze_content
