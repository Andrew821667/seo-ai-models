
from typing import Dict, List, Optional, Union, Tuple
from models.seo_advisor.improved_rank_predictor import ImprovedRankPredictor

class CalibratedRankPredictor(ImprovedRankPredictor):
    """
    Усовершенствованный предиктор ранжирования с калиброванными весами,
    учетом конкурентности ниши и отраслевой специфики
    """
    
    def __init__(self, industry: str = 'default'):
        """
        Инициализация предиктора с улучшенными весами
        
        Args:
            industry: Отрасль для специализированных настроек
        """
        super().__init__(industry)
        
        # Откалиброванные веса с суммой 1.0
        self.feature_weights = {
            'keyword_density': 0.12,       # Снижено с 0.15 - меньше акцента на простую плотность
            'content_length': 0.15,        # Снижено с 0.18 - качество важнее объема
            'readability_score': 0.13,     # Снижено с 0.15
            'meta_tags_score': 0.07,       # Снижено с 0.08
            'header_structure_score': 0.12, # Без изменений
            'multimedia_score': 0.06,      # Снижено с 0.07
            'internal_linking_score': 0.06, # Снижено с 0.07
            'topic_relevance': 0.16,       # Снижено с 0.18
            'semantic_depth': 0.13,        # Снижено с 0.15
            'engagement_potential': 0.10    # Без изменений
        }
        
        # Нормализаторы для входных данных
        self.feature_normalizers = {
            'keyword_density': 0.04,     # Нормализация по максимальной плотности 4%
            'content_length': 2000,      # Нормализация по 2000 слов
            'readability_score': 100,    # Шкала 0-100
            'meta_tags_score': 1.0,      # Шкала 0-1
            'header_structure_score': 1.0, # Шкала 0-1
            'multimedia_score': 1.0,     # Шкала 0-1
            'internal_linking_score': 1.0, # Шкала 0-1
            'topic_relevance': 1.0,      # Шкала 0-1
            'semantic_depth': 1.0,       # Шкала 0-1
            'engagement_potential': 1.0  # Шкала 0-1
        }
        
        # Расширенные отраслевые коэффициенты
        self.industry_adjustments = {
            'electronics': {
                'technical_depth': 1.3,
                'example_weight': 1.2,
                'spec_detail': 1.4,
                'comparison_tables': 1.3,
                'user_reviews': 1.2
            },
            'health': {
                'expert_citations': 1.5,
                'research_references': 1.4,
                'clarity': 1.3,
                'safety_information': 1.5
            },
            'finance': {
                'data_accuracy': 1.6,
                'regulatory_compliance': 1.5,
                'disclaimer_presence': 1.3,
                'current_information': 1.4
            },
            'travel': {
                'location_details': 1.4,
                'practical_info': 1.3,
                'visual_content': 1.5,
                'seasonal_relevance': 1.2
            },
            'education': {
                'learning_structure': 1.4,
                'exercises': 1.3,
                'examples': 1.5,
                'additional_resources': 1.2
            },
            'ecommerce': {
                'product_details': 1.5,
                'pricing_clarity': 1.4,
                'shipping_info': 1.2,
                'review_integration': 1.3
            }
        }
        
        # Коэффициенты конкурентности ниш
        self.competition_factors = {
            'electronics': 1.4,    # Высокая конкуренция
            'health': 1.5,         # Очень высокая (YMYL)
            'finance': 1.6,        # Наивысшая (YMYL)
            'travel': 1.2,         # Средняя
            'education': 1.3,      # Средне-высокая
            'ecommerce': 1.35,     # Высокая
            'local_business': 1.0, # Ниже средней
            'blog': 1.1,           # Низкая
            'news': 1.3,           # Средне-высокая
            'default': 1.2         # По умолчанию
        }
        
        # Пороги качества для предсказания
        self.quality_thresholds = {
            'top10': 0.75,   # Минимальный скор для топ-10
            'top20': 0.60,   # Минимальный скор для топ-20
            'top30': 0.50,   # Минимальный скор для топ-30
            'top50': 0.40    # Минимальный скор для топ-50
        }
        
        # Весовые коэффициенты для ключевых слов по позиции в тексте
        self.keyword_position_weights = {
            'title': 4.0,          # Ключевые слова в заголовке
            'first_paragraph': 2.0, # Ключевые слова в первом абзаце
            'heading': 3.0,        # Ключевые слова в подзаголовках
            'last_paragraph': 1.5,  # Ключевые слова в последнем абзаце
            'body': 1.0            # Ключевые слова в основном тексте
        }

    def calculate_keyword_density(self, text: str, keywords: list) -> float:
        """
        Улучшенный расчет плотности ключевых слов с учетом их расположения
        
        Args:
            text: Анализируемый текст
            keywords: Список ключевых слов
            
        Returns:
            Взвешенная плотность ключевых слов
        """
        # Если текст пустой, возвращаем 0
        if not text or not keywords:
            return 0.0
            
        words = text.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
            
        # Разбиваем текст на секции
        paragraphs = text.split('\n\n')
        title = paragraphs[0] if paragraphs else ""
        first_paragraph = paragraphs[1] if len(paragraphs) > 1 else ""
        last_paragraph = paragraphs[-1] if len(paragraphs) > 1 else ""
        
        # Находим заголовки (строки, начинающиеся с # или ##)
        headings = [p for p in paragraphs if p.strip().startswith('#')]
        
        keyword_count = 0
        weighted_count = 0
        
        for keyword in keywords:
            # Нормализуем ключевое слово
            kw = keyword.lower()
            
            # Подсчет вхождений с учетом положения
            base_count = text.lower().count(kw)
            keyword_count += base_count
            
            # Вхождения в заголовке
            title_count = title.lower().count(kw)
            weighted_count += title_count * self.keyword_position_weights['title']
            
            # Вхождения в первом абзаце
            first_p_count = first_paragraph.lower().count(kw)
            weighted_count += first_p_count * self.keyword_position_weights['first_paragraph']
            
            # Вхождения в подзаголовках
            heading_count = sum(h.lower().count(kw) for h in headings)
            weighted_count += heading_count * self.keyword_position_weights['heading']
            
            # Вхождения в последнем абзаце
            last_p_count = last_paragraph.lower().count(kw)
            weighted_count += last_p_count * self.keyword_position_weights['last_paragraph']
            
            # Вхождения в основном тексте (исключая уже подсчитанные)
            body_count = base_count - title_count - first_p_count - heading_count - last_p_count
            weighted_count += body_count * self.keyword_position_weights['body']
            
            # Бонус для точных совпадений фраз (для мультисловных ключевых слов)
            if ' ' in kw and kw in text.lower():
                weighted_count += text.lower().count(kw) * 2
                
        # Расчет взвешенной плотности
        base_density = keyword_count / total_words
        weighted_density = weighted_count / total_words
        
        # Комбинированная оценка (базовая + взвешенная с коэффициентом)
        return 0.4 * base_density + 0.6 * weighted_density

    def predict_position(self, features: Dict[str, float], text: str = None, keywords: list = None, competitors: list = None) -> Dict:
        """
        Предсказание позиции с учетом конкурентности ниши и локации ключевых слов
        
        Args:
            features: Словарь с метриками контента
            text: Текст для анализа (опционально)
            keywords: Ключевые слова (опционально)
            competitors: Информация о конкурентах (опционально)
            
        Returns:
            Словарь с результатами предсказания
        """
        # Если предоставлены текст и ключевые слова, рассчитываем плотность
        if text and keywords:
            features['keyword_density'] = self.calculate_keyword_density(text, keywords)
        
        # Нормализация входных метрик по шкалам
        normalized_features = {}
        for k, v in features.items():
            if k in self.feature_normalizers and self.feature_normalizers[k] > 0:
                normalized_features[k] = min(max(v / self.feature_normalizers[k], 0), 1)
            else:
                normalized_features[k] = min(max(v, 0), 1)
        
        # Применение весов к нормализованным метрикам
        weighted_scores = {
            k: normalized_features.get(k, 0) * self.feature_weights.get(k, 0) 
            for k in self.feature_weights
        }
        
        # Применение отраслевых коэффициентов
        if self.industry in self.industry_adjustments:
            adj = self.industry_adjustments[self.industry]
            for factor, mult in adj.items():
                if factor in weighted_scores:
                    weighted_scores[factor] *= mult
        
        # Расчет итогового скора
        total_score = sum(weighted_scores.values())
        
        # Получаем коэффициент конкурентности для отрасли
        competition_factor = self.competition_factors.get(
            self.industry, 
            self.competition_factors['default']
        )
        
        # Базовое предсказание позиции с учетом конкурентности
        position = max(1, min(100, 50 * (1 - (total_score / competition_factor) * 1.2)))
        
        # Если предоставлены данные о конкурентах, уточняем позицию
        if competitors:
            position = self._adjust_position_with_competitors(
                total_score, 
                competition_factor, 
                competitors
            )
        
        # Определение вероятных диапазонов позиций
        position_ranges = self._estimate_position_ranges(total_score, competition_factor)
        
        return {
            'position': position,
            'feature_scores': normalized_features,
            'weighted_scores': weighted_scores,
            'total_score': total_score,
            'competition_factor': competition_factor,
            'position_ranges': position_ranges,
            'accuracy_estimate': self._estimate_prediction_accuracy(total_score)
        }
    
    def _adjust_position_with_competitors(
        self, 
        content_score: float, 
        competition_factor: float, 
        competitors: list
    ) -> float:
        """
        Уточнение позиции с учетом данных о конкурентах
        
        Args:
            content_score: Общий скор контента
            competition_factor: Коэффициент конкурентности
            competitors: Список словарей с данными о конкурентах
            
        Returns:
            Уточненная позиция
        """
        # Рассчитываем скоры для конкурентов
        competitor_scores = []
        
        for comp in competitors:
            if 'features' in comp:
                # Нормализуем фичи конкурента
                norm_features = {
                    k: min(max(v / self.feature_normalizers.get(k, 1.0), 0), 1) 
                    for k, v in comp['features'].items()
                }
                
                # Применяем веса
                weighted = {
                    k: norm_features.get(k, 0) * self.feature_weights.get(k, 0) 
                    for k in self.feature_weights if k in norm_features
                }
                
                # Рассчитываем общий скор
                comp_score = sum(weighted.values())
                
                competitor_scores.append({
                    'position': comp.get('position', 0),
                    'score': comp_score
                })
        
        if not competitor_scores:
            return max(1, min(100, 50 * (1 - (content_score / competition_factor) * 1.2)))
        
        # Сортируем конкурентов по скору
        competitor_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Находим позицию относительно конкурентов
        position_estimate = 1
        for comp in competitor_scores:
            if content_score < comp['score']:
                position_estimate += 1
        
        # Комбинируем базовую оценку с относительным рейтингом
        base_position = max(1, min(100, 50 * (1 - (content_score / competition_factor) * 1.2)))
        final_position = (base_position * 0.6) + (position_estimate * 0.4)
        
        return max(1, min(100, final_position))
    
    def _estimate_position_ranges(self, total_score: float, competition_factor: float) -> Dict[str, float]:
        """
        Оценка вероятных диапазонов позиций
        
        Args:
            total_score: Общий скор контента
            competition_factor: Коэффициент конкурентности
            
        Returns:
            Словарь с вероятностями попадания в разные диапазоны
        """
        # Корректируем скор с учетом конкурентности
        adjusted_score = total_score / competition_factor
        
        # Расчет вероятностей по диапазонам
        results = {}
        
        if adjusted_score >= self.quality_thresholds['top10']:
            results['top10_probability'] = min(
                1.0, 
                (adjusted_score - self.quality_thresholds['top10']) * 5 + 0.5
            )
        else:
            results['top10_probability'] = max(
                0.0, 
                (adjusted_score / self.quality_thresholds['top10']) * 0.5
            )
            
        if adjusted_score >= self.quality_thresholds['top20']:
            results['top20_probability'] = min(
                1.0, 
                (adjusted_score - self.quality_thresholds['top20']) * 4 + 0.6
            )
        else:
            results['top20_probability'] = max(
                0.0, 
                (adjusted_score / self.quality_thresholds['top20']) * 0.6
            )
            
        if adjusted_score >= self.quality_thresholds['top30']:
            results['top30_probability'] = min(
                1.0, 
                (adjusted_score - self.quality_thresholds['top30']) * 3 + 0.7
            )
        else:
            results['top30_probability'] = max(
                0.0, 
                (adjusted_score / self.quality_thresholds['top30']) * 0.7
            )
            
        if adjusted_score >= self.quality_thresholds['top50']:
            results['top50_probability'] = min(
                1.0, 
                (adjusted_score - self.quality_thresholds['top50']) * 2 + 0.8
            )
        else:
            results['top50_probability'] = max(
                0.0, 
                (adjusted_score / self.quality_thresholds['top50']) * 0.8
            )
            
        return results
    
    def _estimate_prediction_accuracy(self, total_score: float) -> float:
        """
        Оценка точности предсказания
        
        Args:
            total_score: Общий скор контента
            
        Returns:
            Оценка точности от 0 до 1
        """
        # Для очень низких и очень высоких скоров точность обычно выше
        if total_score < 0.3:
            return 0.8  # Уверены в плохих результатах
        elif total_score > 0.8:
            return 0.8  # Уверены в хороших результатах
        else:
            # Для средних значений точность ниже
            return 0.6 + (total_score - 0.3) * 0.2
    
    def generate_recommendations(self, features: Dict[str, float], text: str = None) -> Dict[str, List[str]]:
        """
        Генерация конкретных рекомендаций по улучшению контента
        
        Args:
            features: Словарь с метриками контента
            text: Исходный текст для анализа (опционально)
            
        Returns:
            Словарь с рекомендациями по каждой метрике
        """
        recommendations = {}
        
        # Нормализуем метрики для оценки
        normalized_features = {
            k: min(max(v / self.feature_normalizers.get(k, 1.0), 0), 1) 
            for k, v in features.items()
        }
        
        # Пороги для рекомендаций
        thresholds = {
            'keyword_density': {'critical': 0.3, 'medium': 0.6},
            'content_length': {'critical': 0.3, 'medium': 0.6},
            'readability_score': {'critical': 0.3, 'medium': 0.6},
            'meta_tags_score': {'critical': 0.3, 'medium': 0.6},
            'header_structure_score': {'critical': 0.3, 'medium': 0.6},
            'multimedia_score': {'critical': 0.3, 'medium': 0.6},
            'internal_linking_score': {'critical': 0.3, 'medium': 0.6},
            'topic_relevance': {'critical': 0.4, 'medium': 0.7},
            'semantic_depth': {'critical': 0.4, 'medium': 0.7},
            'engagement_potential': {'critical': 0.3, 'medium': 0.6}
        }
        
        # Детальные рекомендации по каждой метрике
        recommendation_templates = {
            'keyword_density': {
                'critical': [
                    "КРИТИЧНО: Увеличьте плотность ключевых слов до 1-2%",
                    "Добавьте ключевые слова в заголовки H1, H2 и H3",
                    "Включите ключевые слова в первый и последний абзацы"
                ],
                'medium': [
                    "Добавьте больше LSI и семантически связанных слов",
                    "Используйте ключевые слова в подписях к изображениям",
                    "Проверьте разнообразие ключевых слов"
                ],
                'good': [
                    "Проверьте естественность использования ключевых слов",
                    "Убедитесь, что все ключевые термины объяснены"
                ]
            },
            'content_length': {
                'critical': [
                    "КРИТИЧНО: Увеличьте длину текста минимум в 2-3 раза",
                    "Добавьте дополнительные разделы по каждому аспекту темы",
                    "Включите подробные объяснения и примеры"
                ],
                'medium': [
                    "Расширьте контент до 1500+ слов",
                    "Добавьте ответы на часто задаваемые вопросы",
                    "Включите исследования и статистику по теме"
                ],
                'good': [
                    "Обновите контент свежими данными",
                    "Добавьте экспертные мнения для усиления"
                ]
            },
            'readability_score': {
                'critical': [
                    "КРИТИЧНО: Упростите предложения и используйте более понятные термины",
                    "Разбейте текст на короткие абзацы по 2-3 предложения",
                    "Замените сложные термины на более простые аналоги"
                ],
                'medium': [
                    "Добавьте подзаголовки каждые 300 слов",
                    "Используйте маркированные списки для перечислений",
                    "Сократите длину предложений до 15-20 слов"
                ],
                'good': [
                    "Проверьте текст на наличие жаргона",
                    "Добавьте определения для сложных терминов"
                ]
            }
        }
        
        # Добавляем отраслевую специфику
        industry_specific_recommendations = self._get_industry_specific_recommendations()
        
        # Формируем рекомендации
        for feature, value in normalized_features.items():
            if feature in thresholds:
                level = 'good'
                if value < thresholds[feature]['critical']:
                    level = 'critical'
                elif value < thresholds[feature]['medium']:
                    level = 'medium'
                
                # Базовые рекомендации
                feature_recommendations = recommendation_templates.get(feature, {}).get(level, [])
                if not feature_recommendations:
                    feature_recommendations = [f"Улучшите показатель {feature}"]
                
                # Добавляем отраслевые рекомендации, если есть
                if self.industry in industry_specific_recommendations:
                    industry_recs = industry_specific_recommendations[self.industry].get(feature, [])
                    if industry_recs:
                        feature_recommendations.extend(industry_recs)
                
                recommendations[feature] = feature_recommendations
        
        # Анализ текста, если предоставлен
        if text:
            text_recommendations = self._analyze_text_for_recommendations(text, normalized_features)
            for feature, recs in text_recommendations.items():
                if feature in recommendations:
                    recommendations[feature].extend(recs)
                else:
                    recommendations[feature] = recs
        
        return recommendations
    
    def _get_industry_specific_recommendations(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Получение отраслевых рекомендаций
        
        Returns:
            Словарь с рекомендациями по отраслям и метрикам
        """
        return {
            'electronics': {
                'content_length': [
                    "Добавьте раздел с техническими спецификациями",
                    "Включите сравнительные таблицы с конкурентами"
                ],
                'multimedia_score': [
                    "Добавьте фотографии продукта с разных ракурсов",
                    "Включите диаграммы с основными характеристиками"
                ]
            },
            'health': {
                'content_length': [
                    "Добавьте раздел с научными исследованиями",
                    "Включите мнения медицинских экспертов"
                ],
                'topic_relevance': [
                    "Цитируйте авторитетные медицинские источники",
                    "Добавьте предупреждения и медицинские оговорки"
                ]
            },
            'finance': {
                'content_length': [
                    "Добавьте кейсы и примеры расчетов",
                    "Включите раздел с аналитикой рынка"
                ],
                'readability_score': [
                    "Объясните финансовые термины простым языком",
                    "Добавьте примеры финансовых расчетов"
                ]
            },
            'travel': {
                'multimedia_score': [
                    "Добавьте больше фотографий мест и достопримечательностей",
                    "Включите карты и маршруты"
                ],
                'content_length': [
                    "Добавьте практические советы для путешественников",
                    "Включите информацию о сезонности и лучшем времени для посещения"
                ]
            }
        }
    
    def _analyze_text_for_recommendations(self, text: str, features: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Анализ текста для генерации конкретных рекомендаций
        
        Args:
            text: Текст контента
            features: Нормализованные метрики
            
        Returns:
            Словарь с рекомендациями на основе анализа текста
        """
        recommendations = {}
        
        # Анализ заголовков
        paragraphs = text.split('\n\n')
        headings = [p for p in paragraphs if p.strip().startswith('#')]
        
        if len(headings) < 3 and features.get('header_structure_score', 1) < 0.7:
            if 'header_structure_score' not in recommendations:
                recommendations['header_structure_score'] = []
            recommendations['header_structure_score'].append(
                "Добавьте больше подзаголовков (H2, H3) для структурирования текста"
            )
        
        # Анализ предложений
        sentences = [s for p in paragraphs for s in p.split('. ') if s]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length > 20 and features.get('readability_score', 1) < 0.7:
            if 'readability_score' not in recommendations:
                recommendations['readability_score'] = []
            recommendations['readability_score'].append(
                f"Сократите среднюю длину предложений (сейчас ~{avg_sentence_length:.1f} слов)"
            )
        
        # Анализ абзацев
        text_paragraphs = [p for p in paragraphs if p and not p.strip().startswith('#')]
        long_paragraphs = [p for p in text_paragraphs if len(p.split()) > 100]
        
        if long_paragraphs and features.get('readability_score', 1) < 0.8:
            if 'readability_score' not in recommendations:
                recommendations['readability_score'] = []
            recommendations['readability_score'].append(
                f"Разбейте {len(long_paragraphs)} длинных абзацев на более короткие части"
            )
        
        return recommendations
