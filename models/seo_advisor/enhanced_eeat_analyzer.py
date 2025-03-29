from typing import Dict, List, Optional, Union, Any, Tuple
import re
from collections import Counter
import numpy as np
import random
try:
    from sklearn.ensemble import RandomForestRegressor
    import joblib
except ImportError:
    print("sklearn не установлен. Функции машинного обучения будут недоступны.")
import os
from pathlib import Path
try:
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse
except ImportError:
    print("requests или beautifulsoup4 не установлены. Анализ внешних ссылок будет ограничен.")
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    print("nltk не установлен. Семантический анализ будет ограничен.")
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка необходимых ресурсов NLTK
def download_nltk_resources():
    """Загрузка всех необходимых ресурсов NLTK"""
    try:
        resources = ['punkt', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)
        print("✅ NLTK ресурсы успешно загружены")
    except Exception as e:
        logger.warning(f"Не удалось загрузить ресурсы NLTK: {str(e)}")

# Вызываем функцию загрузки ресурсов при импорте модуля
download_nltk_resources()

class EnhancedEEATAnalyzer:
    """
    Усовершенствованный анализатор E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
    с поддержкой машинного обучения, анализа внешних ссылок и семантического анализа
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Инициализация базовых компонентов
        self._initialize_markers()
        self._initialize_industry_markers()
        self._initialize_patterns()
        self._initialize_ymyl_industries()
        
        # Инициализация модели машинного обучения
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Модель машинного обучения загружена из {model_path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель из {model_path}: {str(e)}")
        
        # Инициализация компонентов семантического анализа
        try:
            self.stop_words = set(stopwords.words('russian')).union(set(stopwords.words('english')))
        except:
            self.stop_words = set(['и', 'в', 'не', 'на', 'с', 'по', 'у', 'к', 'о', 'за', 'а', 'the', 'of', 'to', 'in', 'and', 'a', 'is', 'for', 'with'])
            logger.warning("Используем ограниченный набор стоп-слов из-за проблем с NLTK.")
        
        # Кэш для внешних анализов
        self.external_links_cache = {}

    def _initialize_markers(self):
        """Инициализация маркеров E-E-A-T"""
        # Маркеры опыта и экспертизы
        self.expertise_markers = [
            'опыт', 'эксперт', 'специалист', 'профессионал', 'квалификация',
            'сертифицированный', 'компетентный', 'практик', 'исследователь', 'аналитик',
            'образование', 'обучение', 'диплом', 'степень', 'награда', 'стаж', 'практика',
            'достижение', 'сертификат', 'знания', 'навык', 'умение', 'подготовка',
            'лет опыта', 'карьера', 'магистр', 'бакалавр', 'PhD', 'профессор', 'доцент', 'академик'
        ]
        
        # Маркеры авторитетности
        self.authority_markers = [
            'исследование', 'статистика', 'данные', 'доказано', 'согласно', 
            'по мнению экспертов', 'научно', 'источник', 'цитата', 'ссылка',
            'авторитет', 'эксперты рекомендуют', 'в соответствии с', 'рецензируемый',
            'признанный', 'известный', 'уважаемый', 'влиятельный', 'популярный',
            'научная статья', 'журнал', 'публикация', 'библиография', 'рейтинг',
            'первое место', 'лидер рынка', 'официальные данные', 'госстандарт'
        ]
        
        # Маркеры доверия
        self.trust_markers = [
            'достоверный', 'проверенный', 'надежный', 'точный', 'подтвержденный',
            'официальный', 'гарантированный', 'безопасный', 'проверка фактов', 'прозрачность',
            'методология', 'метод', 'данные показывают', 'доказательство', 'подтверждено',
            'публикация', 'обновлено', 'раскрытие информации', 'отказ от ответственности', 'дисклеймер',
            'источники данных', 'обновление', 'актуальность', 'текущий', 'сравнение',
            'анализ', 'критерии', 'протокол', 'этика', 'независимый', 'объективный',
            'рецензирование', 'верификация', 'аудит', 'сертификация', 'стандарт',
            'факт', 'измеримый', 'повторяемый', 'проверяемый', 'сертифицирован'
        ]
    
    def _initialize_industry_markers(self):
        """Инициализация отраслевых маркеров"""
        self.industry_markers = {
            'finance': {
                'expertise': [
                    'финансовый аналитик', 'экономист', 'банкир', 'финансовый советник', 'аудитор', 
                    'брокер', 'трейдер', 'налоговый консультант', 'бухгалтер', 'финансовый директор',
                    'CFA', 'MBA', 'кандидат экономических наук', 'доктор экономических наук'
                ],
                'authority': [
                    'центральный банк', 'минфин', 'регулятор', 'налоговая служба', 'финансовый отчет',
                    'биржа', 'фондовый рынок', 'Moody\'s', 'S&P', 'Fitch', 'Bloomberg', 'Reuters',
                    'Financial Times', 'Wall Street Journal', 'Банк России'
                ],
                'trust': [
                    'лицензия', 'сертификат', 'регулируемый', 'финансовая отчетность', 'документация',
                    'МСФО', 'аудиторское заключение', 'финансовый мониторинг', 'комплаенс',
                    'прозрачность операций', 'страхование вкладов', 'гарантия возврата'
                ]
            },
            'health': {
                'expertise': [
                    'врач', 'медицинский', 'фармацевт', 'терапевт', 'хирург', 'клиника', 'больница',
                    'доктор медицинских наук', 'кандидат медицинских наук', 'медицинское образование',
                    'ординатура', 'интернатура', 'стаж работы в медицине', 'профессор медицины'
                ],
                'authority': [
                    'министерство здравоохранения', 'воз', 'медицинский журнал', 'клинические испытания',
                    'lancet', 'nejm', 'jama', 'bmj', 'pubmed', 'medline', 'cochrane',
                    'рандомизированное контролируемое исследование', 'мета-анализ', 'систематический обзор'
                ],
                'trust': [
                    'пациент', 'диагноз', 'лечение', 'протокол лечения', 'медицинская этика', 'показания',
                    'медикаментозная терапия', 'хирургическое вмешательство', 'диагностика',
                    'медицинская лицензия', 'сертификат врача', 'консилиум', 'второе мнение'
                ]
            },
            'legal': {
                'expertise': [
                    'юрист', 'адвокат', 'нотариус', 'прокурор', 'судья', 'законодательство', 'кодекс',
                    'юридическое образование', 'юридический стаж', 'специалист по праву',
                    'кандидат юридических наук', 'доктор юридических наук', 'магистр права'
                ],
                'authority': [
                    'суд', 'кодекс', 'закон', 'правовой акт', 'постановление', 'решение суда',
                    'верховный суд', 'конституционный суд', 'пленум верховного суда',
                    'официальный сайт суда', 'государственная дума', 'министерство юстиции'
                ],
                'trust': [
                    'правовая норма', 'законодательный акт', 'нормативный', 'юридическая сила', 'легальный',
                    'правоприменительная практика', 'судебный прецедент', 'юридический прецедент',
                    'адвокатская тайна', 'конфиденциальность', 'презумпция невиновности'
                ]
            },
            'tech': {
                'expertise': [
                    'разработчик', 'программист', 'инженер', 'технолог', 'айти-специалист', 'дизайнер',
                    'системный администратор', 'DevOps-инженер', 'data scientist', 'UX-дизайнер',
                    'frontend-разработчик', 'backend-разработчик', 'full-stack', 'CTO'
                ],
                'authority': [
                    'техническая документация', 'спецификация', 'стандарт', 'api', 'протокол',
                    'github', 'stackoverflow', 'ieee', 'w3c', 'ietf', 'iso', 'релиз', 
                    'белая книга', 'техническое руководство', 'документация разработчика'
                ],
                'trust': [
                    'версия', 'релиз', 'патч', 'обновление', 'баг-фикс', 'тестирование', 'валидация',
                    'code review', 'peer review', 'open source', 'unit-тесты', 'qa', 'сертификация',
                    'совместимость', 'безопасность данных', 'шифрование', 'резервное копирование'
                ]
            },
            'ecommerce': {
                'expertise': [
                    'ритейлер', 'продавец', 'маркетолог', 'специалист по продажам', 'менеджер по продукту',
                    'категорийный менеджер', 'специалист по логистике', 'мерчендайзер', 'закупщик'
                ],
                'authority': [
                    'отзывы клиентов', 'рейтинг товаров', 'сравнение цен', 'обзор товаров',
                    'тестирование продукта', 'сертификация продукции', 'награды', 'признание бренда'
                ],
                'trust': [
                    'гарантия', 'возврат', 'доставка', 'оплата', 'безопасная сделка', 'политика конфиденциальности',
                    'сертификат качества', 'официальный дистрибьютор', 'авторизованный продавец'
                ]
            }
        }
    
    def _initialize_patterns(self):
        """Инициализация шаблонов регулярных выражений"""
        self.special_trust_patterns = [
            r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # Даты в различных форматах
            r'\d{4} год',  # Годы
            r'\d{1,2} [а-я]+ \d{4}',  # Даты в текстовом формате
            r'\d+%',  # Проценты
            r'\d+\s*(?:млн|млрд|тыс)',  # Числовые значения с единицами измерения
            r'(?:согласно|по данным|по информации)[^.!?]*источник[^.!?]*',  # Ссылки на источники
            r'автор[^.!?]*опыт[^.!?]*',  # Информация об опыте автора
            r'раскрытие информации[^.!?]*',  # Секции с раскрытием информации
            r'методология[^.!?]*',  # Секции с методологией
            r'об авторе[^.!?]*',  # Информация об авторе
            r'ссылк[а-я] на исследовани[а-я]',  # Ссылки на исследования
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URL
            r'(?:по состоянию на|актуально на)[^.!?]*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # Актуальность данных
            r'(?:рецензирован|проверен)[^.!?]*эксперт',  # Рецензирование экспертами
            r'©\s*\d{4}',  # Копирайт с годом
            r'все права защищены',  # Права
            r'зарегистрированный товарный знак',  # Товарные знаки
            r'(?:обновлено|последнее обновление)[^.!?]*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # Дата обновления
            r'№\s*\d+',  # Номера лицензий, сертификатов
            r'лицензия[^.!?]*\d+',  # Номера лицензий
            r'ГОСТ[^.!?]*\d+',  # ГОСТы
            r'стандарт[^.!?]*\d+',  # Стандарты с номерами
            r'исследовани[а-я][^.!?]*\d+[^.!?]*участник',  # Описания исследований с числом участников
            r'выборк[а-я][^.!?]*\d+[^.!?]*человек'  # Описания выборки
        ]
        
        # Шаблоны для извлечения цитат
        self.citation_patterns = [
            r'"[^"]{10,300}"',  # Текст в двойных кавычках
            r'«[^»]{10,300}»',  # Текст в елочках
            r'[^.!?]*по словам[^.!?]*,[^.!?]*\.',  # "По словам ..."
            r'[^.!?]*как отметил[^.!?]*,[^.!?]*\.',  # "Как отметил ..."
            r'[^.!?]*заявил[^.!?]*,[^.!?]*\.'  # "... заявил ..."
        ]
        
        # Шаблоны для извлечения ссылок
        self.link_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URL
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'  # www без http
        ]
    
    def _initialize_ymyl_industries(self):
        """Инициализация YMYL отраслей"""
        self.ymyl_industries = {
            'finance': True,
            'health': True,
            'legal': True,
            'insurance': True,
            'medical': True,
            'crypto': True,
            'investment': True,
            'taxes': True,
            'retirement': True,
            'estate_planning': True,
            'medicine': True,
            'pharmaceuticals': True,
            'supplements': True,
            'mental_health': True,
            'diet': True,
            'law': True,
            'government': True
        }
    
    def analyze(
        self, 
        text: str, 
        industry: str = 'default', 
        urls: Optional[List[str]] = None,
        external_analysis: bool = False
    ) -> Dict[str, Union[float, Dict, List]]:
        """
        Расширенная оценка E-E-A-T сигналов в тексте
        
        Args:
            text: Анализируемый текст
            industry: Отрасль контента
            urls: Список URL, упомянутых в тексте (опционально)
            external_analysis: Выполнять ли анализ внешних ссылок
            
        Returns:
            Словарь с оценками и рекомендациями
        """
        text_lower = text.lower()
        is_ymyl = self.ymyl_industries.get(industry, False)
        
        # Получение маркеров, специфичных для отрасли
        industry_specific_markers = self.industry_markers.get(industry, {})
        
        # 1. Оценка опыта и экспертизы с учетом отрасли
        expertise_markers = self.expertise_markers[:]
        if 'expertise' in industry_specific_markers:
            expertise_markers.extend(industry_specific_markers['expertise'])
        expertise_score = self._evaluate_markers(text_lower, expertise_markers)
        
        # 2. Оценка авторитетности с учетом отрасли
        authority_markers = self.authority_markers[:]
        if 'authority' in industry_specific_markers:
            authority_markers.extend(industry_specific_markers['authority'])
        authority_score = self._evaluate_markers(text_lower, authority_markers)
        
        # 3. Оценка доверия с учетом отрасли
        trust_markers = self.trust_markers[:]
        if 'trust' in industry_specific_markers:
            trust_markers.extend(industry_specific_markers['trust'])
        trust_score = self._evaluate_trust(text, text_lower, trust_markers)
        
        # 4. Расширенный анализ структуры и ссылок
        structural_analysis = self._analyze_structure(text)
        structural_score = structural_analysis['score']
        
        # 5. Семантический анализ для выявления тематической экспертизы
        semantic_analysis = self._perform_semantic_analysis(text, industry)
        semantic_coherence_score = semantic_analysis['coherence_score']
        
        # 6. Анализ цитирования и ссылок
        citation_analysis = self._analyze_citations(text)
        citation_score = citation_analysis['score']
        
        # 7. Анализ внешних ссылок, если указано
        external_links_score = 0.0
        external_links_analysis = {'score': 0.0, 'details': {}}
        
        if external_analysis and urls:
            external_links_analysis = self._analyze_external_links(urls, industry)
            external_links_score = external_links_analysis['score']
        
        # 8. Интеграция всех компонентов и расчет финального скора
        # Для YMYL отраслей увеличиваем вес доверия и авторитетности
        eeat_scores = {
            'expertise': expertise_score,
            'authority': authority_score,
            'trust': trust_score,
            'structure': structural_score,
            'semantics': semantic_coherence_score,
            'citations': citation_score,
            'external_links': external_links_score
        }
        
        # Используем модель машинного обучения, если доступна
        if self.model:
            try:
                score_vector = np.array([
                    expertise_score, authority_score, trust_score, 
                    structural_score, semantic_coherence_score, 
                    citation_score, external_links_score,
                    1 if is_ymyl else 0  # YMYL фактор
                ]).reshape(1, -1)
                
                overall_eeat_score = float(self.model.predict(score_vector)[0])
                logger.info(f"Оценка E-E-A-T с использованием модели МО: {overall_eeat_score:.3f}")
            except Exception as e:
                logger.warning(f"Ошибка при использовании модели: {str(e)}. Используем взвешенную сумму.")
                overall_eeat_score = self._calculate_weighted_score(eeat_scores, is_ymyl)
        else:
            overall_eeat_score = self._calculate_weighted_score(eeat_scores, is_ymyl)
        
        # 9. Компоненты и подробности анализа
        component_details = {
            'expertise': {
                'score': expertise_score,
                'found_markers': self._get_found_markers(text_lower, expertise_markers),
                'count': self._count_markers(text_lower, expertise_markers),
                'status': self._get_status(expertise_score)
            },
            'authority': {
                'score': authority_score,
                'found_markers': self._get_found_markers(text_lower, authority_markers),
                'count': self._count_markers(text_lower, authority_markers),
                'status': self._get_status(authority_score)
            },
            'trust': {
                'score': trust_score,
                'found_markers': self._get_found_markers(text_lower, trust_markers),
                'count': self._count_markers(text_lower, trust_markers),
                'special_markers': self._count_special_markers(text),
                'status': self._get_status(trust_score)
            },
            'structure': structural_analysis,
            'semantics': semantic_analysis,
            'citations': citation_analysis,
            'external_links': external_links_analysis
        }
        
        # 10. Формирование рекомендаций с учетом всех аспектов
        recommendations = self._generate_recommendations(
            component_details, is_ymyl, industry, overall_eeat_score
        )
        
        return {
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'semantic_coherence_score': semantic_coherence_score,
            'citation_score': citation_score,
            'external_links_score': external_links_score,
            'overall_eeat_score': overall_eeat_score,
            'recommendations': recommendations,
            'component_details': component_details,
            'ymyl_status': is_ymyl,
            'industry': industry
        }

    def _calculate_weighted_score(self, scores: Dict[str, float], is_ymyl: bool) -> float:
        """Расчет взвешенного скора на основе компонентов"""
        if is_ymyl:
            weights = {
                'expertise': 0.20,
                'authority': 0.25,
                'trust': 0.30,
                'structure': 0.05,
                'semantics': 0.10,
                'citations': 0.05,
                'external_links': 0.05
            }
        else:
            weights = {
                'expertise': 0.25,
                'authority': 0.20,
                'trust': 0.20,
                'structure': 0.10,
                'semantics': 0.15,
                'citations': 0.05,
                'external_links': 0.05
            }
        
        weighted_sum = sum(scores[k] * weights[k] for k in scores)
        return min(1.0, max(0.0, weighted_sum))
    
    def _evaluate_markers(self, text: str, markers: List[str]) -> float:
        """Оценка наличия маркеров в тексте"""
        total_markers = len(markers)
        found_markers = sum(1 for marker in markers if marker in text)
        
        # Оценка на основе количества найденных маркеров
        marker_score = min(found_markers / (total_markers / 2), 1.0)
        
        # Оцениваем распределение маркеров
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        distribution_score = 0.0
        
        if paragraphs:
            markers_per_paragraph = []
            for paragraph in paragraphs:
                markers_found = sum(1 for marker in markers if marker in paragraph.lower())
                markers_per_paragraph.append(markers_found)
            
            # Оценка распределения маркеров по параграфам
            if sum(markers_per_paragraph) > 0:
                distribution_score = min(1.0, len([m for m in markers_per_paragraph if m > 0]) / len(paragraphs))
        
        # Комбинированная оценка
        return 0.7 * marker_score + 0.3 * distribution_score
    
    def _evaluate_trust(self, text: str, text_lower: str, trust_markers: List[str]) -> float:
        """Улучшенная оценка доверия"""
        # Базовая оценка на основе маркеров
        base_trust_score = self._evaluate_markers(text_lower, trust_markers)
        
        # Поиск специальных маркеров доверия с помощью регулярных выражений
        special_markers_count = 0
        for pattern in self.special_trust_patterns:
            matches = re.findall(pattern, text)
            special_markers_count += len(matches)
        
        # Оценка наличия специальных маркеров доверия
        special_score = min(special_markers_count / 10, 1.0)
        
        # Проверка наличия важных секций (методология, об авторе, раскрытие информации)
        important_sections_score = 0.0
        if "методология" in text_lower or "источники" in text_lower:
            important_sections_score += 0.4
        
        if "об авторе" in text_lower or "автор:" in text_lower:
            important_sections_score += 0.3
        
        if "раскрытие информации" in text_lower or "дисклеймер" in text_lower:
            important_sections_score += 0.3
        
        # Поиск дат в тексте
        date_patterns = [r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', r'\d{4} год', r'\d{1,2} [а-я]+ \d{4}']
        dates_found = False
        for pattern in date_patterns:
            if re.search(pattern, text):
                dates_found = True
                break
        
        date_score = 0.2 if dates_found else 0.0
        
        # Комбинированная оценка доверия
        trust_score = (
            base_trust_score * 0.4 +
            special_score * 0.3 +
            important_sections_score * 0.2 +
            date_score * 0.1
        )
        
        return trust_score
    
    def _analyze_structure(self, text: str) -> Dict[str, Union[float, Dict]]:
        """Расширенный анализ структуры текста"""
        # Структурные элементы
        elements = self._count_structural_elements(text)
        
        # Оценка наличия цитат
        quotes = elements['quotes']
        quotes_score = min(quotes / 8, 1.0)  # Учитываем открывающие и закрывающие кавычки
        
        # Оценка наличия ссылок на источники
        sources_ref = elements['source_references']
        sources_score = min(sources_ref / 4, 1.0)
        
        # Оценка наличия структурированных данных (списки)
        lists = elements['lists']
        lists_score = min(lists / 8, 1.0)
        
        # Оценка наличия подзаголовков
        headers = elements['headers']
        headers_score = min(headers / 5, 1.0)
        
        # Оценка наличия таблиц
        tables = elements['tables']
        tables_score = min(tables, 1.0)
        
        # Взвешенная оценка структуры
        score = (
            quotes_score * 0.2 +
            sources_score * 0.3 +
            lists_score * 0.2 +
            headers_score * 0.2 +
            tables_score * 0.1
        )
        
        return {
            'score': score,
            'elements': elements,
            'component_scores': {
                'quotes': quotes_score,
                'sources': sources_score,
                'lists': lists_score,
                'headers': headers_score,
                'tables': tables_score
            },
            'status': self._get_status(score)
        }
    
    def _count_structural_elements(self, text: str) -> Dict[str, int]:
        """Расширенный подсчет структурных элементов в тексте"""
        elements = {
            'quotes': text.count('"') + text.count("'") + text.count("«") + text.count("»"),
            'lists': text.count('-') + text.count('*') + text.count('•') + text.count('1.'),
            'headers': text.count('#') + text.count('##') + text.count('###'),
            'tables': text.count('|') // 10,
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
            'source_references': (
                text.lower().count('источник') + 
                text.lower().count('ссылк') + 
                text.lower().count('http') + 
                text.lower().count('www')
            ),
            'bullet_points': text.count('•') + text.count('-') + text.count('*'),
            'numbered_items': sum(1 for line in text.split('\n') if re.match(r'^\s*\d+\.', line))
        }
        
        # Дополнительно ищем возможные URL и ссылки
        urls = []
        for pattern in self.link_patterns:
            matches = re.findall(pattern, text)
            urls.extend(matches)
        elements['urls'] = len(urls)
        
        # Ищем цитаты
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        elements['citations'] = len(citations)
        
        return elements
    
    def _analyze_citations(self, text: str) -> Dict[str, Union[float, Dict, List]]:
        """Анализ цитат и ссылок на экспертов"""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        # Ограничиваем длину цитат для вывода
        formatted_citations = []
        for citation in citations:
            if len(citation) > 100:
                citation = citation[:97] + "..."
            formatted_citations.append(citation)
        
        # Оценка качества цитирования
        citation_count = len(citations)
        citation_score = min(citation_count / 5, 1.0)
        
        # Анализ контекста цитат (проверяем наличие экспертов в контексте цитат)
        expert_citations = 0
        expert_words = ['эксперт', 'специалист', 'исследователь', 'профессор', 'доктор', 'кандидат']
        
        for citation in citations:
            for word in expert_words:
                if word in citation.lower():
                    expert_citations += 1
                    break
        
        # Увеличиваем скор, если есть цитаты экспертов
        expert_citation_ratio = expert_citations / max(1, citation_count)
        enhanced_score = citation_score * (1 + expert_citation_ratio * 0.5)
        final_score = min(enhanced_score, 1.0)
        
        return {
            'score': final_score,
            'citation_count': citation_count,
            'expert_citations': expert_citations,
            'examples': formatted_citations[:5],  # Ограничиваем до 5 примеров
            'status': self._get_status(final_score)
        }
    
    def _analyze_external_links(self, urls: List[str], industry: str) -> Dict[str, Union[float, Dict]]:
        """Анализ внешних ссылок для оценки авторитетности"""
        if not urls:
            return {'score': 0.0, 'details': {}, 'status': 'Нет данных'}
        
        # Кэширование результатов для ускорения повторных анализов
        cached_results = {url: self.external_links_cache.get(url) for url in urls if url in self.external_links_cache}
        urls_to_analyze = [url for url in urls if url not in self.external_links_cache]
        
        # Определение авторитетных доменов для каждой отрасли
        authority_domains = {
            'finance': ['cbr.ru', 'minfin.ru', 'bloomberg.com', 'ft.com', 'wsj.com', 'reuters.com', 'moex.com'],
            'health': ['who.int', 'minzdrav.ru', 'nih.gov', 'nejm.org', 'thelancet.com', 'bmj.com', 'pubmed.gov'],
            'legal': ['pravo.gov.ru', 'consultant.ru', 'garant.ru', 'vsrf.ru', 'ksrf.ru'],
            'tech': ['github.com', 'stackoverflow.com', 'ieee.org', 'w3.org', 'ietf.org', 'microsoft.com', 'google.com'],
            'default': ['wikipedia.org', 'edu', 'gov', 'ac.uk', 'github.com', 'scholar.google.com']
        }
        
        industry_domains = authority_domains.get(industry, authority_domains['default'])
        
        # Анализ доменов
        results = {}
        authority_count = 0
        total_analyzed = 0
        
        # Добавляем кэшированные результаты
        for url, cached_data in cached_results.items():
            if cached_data:
                results[url] = cached_data
                if cached_data.get('is_authority', False):
                    authority_count += 1
                total_analyzed += 1
        
        # Анализируем новые URL
        for url in urls_to_analyze:
            try:
                # Базовый анализ домена
                domain = urlparse(url).netloc
                is_edu_gov = any(ext in domain for ext in ['.edu', '.gov', '.org', '.ac.uk'])
                is_authority = any(auth_domain in domain for auth_domain in industry_domains) or is_edu_gov
                
                # Упрощенный анализ без HTTP-запросов для Colab
                title = 'Без заголовка (требуется HTTP-запрос)'
                
                # Сохраняем результат
                result = {
                    'domain': domain,
                    'is_authority': is_authority,
                    'is_edu_gov': is_edu_gov,
                    'title': title
                }
                
                # Кэшируем результат
                self.external_links_cache[url] = result
                results[url] = result
                
                if is_authority:
                    authority_count += 1
                total_analyzed += 1
                
            except Exception as e:
                logger.warning(f"Ошибка при анализе ссылки {url}: {str(e)}")
        
        # Расчет итогового скора
        authority_ratio = authority_count / max(1, total_analyzed)
        diversity_factor = min(total_analyzed / 3, 1.0)  # Поощряем разнообразие источников
        
        final_score = authority_ratio * 0.7 + diversity_factor * 0.3
        
        return {
            'score': final_score,
            'details': {
                'total_links': len(urls),
                'analyzed_links': total_analyzed,
                'authority_links': authority_count,
                'authority_ratio': authority_ratio,
                'link_details': results
            },
            'status': self._get_status(final_score)
        }
    
    def _perform_semantic_analysis(self, text: str, industry: str) -> Dict[str, Union[float, Dict]]:
        """Семантический анализ текста для оценки тематической связности"""
        # Токенизация и удаление стоп-слов
        try:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [w for w in tokens if w.isalnum() and w not in self.stop_words]
            
            # Если текст слишком короткий, возвращаем низкую оценку
            if len(filtered_tokens) < 10:
                return {
                    'coherence_score': 0.2,
                    'status': 'Недостаточно контента для анализа',
                    'details': {'token_count': len(filtered_tokens)}
                }
            
            # Вычисление частоты слов
            word_freq = Counter(filtered_tokens)
            
            # Извлечение ключевых слов (верхние 5%)
            top_n = max(5, int(len(word_freq) * 0.05))
            top_words = [word for word, _ in word_freq.most_common(top_n)]
            
            # Оценка тематической концентрации (доля текста, занимаемая ключевыми словами)
            top_words_count = sum(word_freq[word] for word in top_words)
            concentration = top_words_count / len(filtered_tokens)
            
            # Оценка связности (как часто ключевые слова встречаются рядом)
            coherence = 0.0
            coherence_pairs = 0
            
            # Проверяем, как часто ключевые слова встречаются в одном окне
            window_size = 5
            for i in range(len(filtered_tokens) - window_size + 1):
                window = filtered_tokens[i:i + window_size]
                key_words_in_window = sum(1 for w in window if w in top_words)
                if key_words_in_window >= 2:
                    coherence_pairs += 1
            
            # Нормализация
            max_possible_windows = max(1, len(filtered_tokens) - window_size + 1)
            coherence = min(coherence_pairs / max_possible_windows * 3, 1.0)  # Множитель 3 для усиления сигнала
            
            # Итоговая оценка с учетом концентрации и связности
            coherence_score = 0.6 * concentration + 0.4 * coherence
            
            return {
                'coherence_score': coherence_score,
                'details': {
                    'token_count': len(filtered_tokens),
                    'unique_tokens': len(word_freq),
                    'top_words': top_words,
                    'concentration': concentration,
                    'coherence': coherence
                },
                'status': self._get_status(coherence_score)
            }
            
        except Exception as e:
            logger.error(f"Ошибка при семантическом анализе: {str(e)}")
            return {
                'coherence_score': 0.0,
                'status': 'Ошибка анализа',
                'details': {'error': str(e)}
            }
    
    def _get_found_markers(self, text: str, markers: List[str]) -> List[str]:
        """Получение списка найденных маркеров"""
        return [marker for marker in markers if marker in text]
    
    def _count_markers(self, text: str, markers: List[str]) -> Dict[str, int]:
        """Подсчет количества каждого маркера в тексте"""
        return {marker: text.count(marker) for marker in markers if marker in text}
    
    def _count_special_markers(self, text: str) -> Dict[str, int]:
        """Подсчет специальных маркеров в тексте"""
        results = {}
        for pattern in self.special_trust_patterns:
            matches = re.findall(pattern, text)
            if matches:
                pattern_name = pattern.replace('\\', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '')[:20]
                results[pattern_name] = len(matches)
        return results
    
    def _get_status(self, score: float) -> str:
        """Получение текстового статуса на основе оценки"""
        if score < 0.3:
            return "Требует существенного улучшения"
        elif score < 0.5:
            return "Требует улучшения"
        elif score < 0.7:
            return "Удовлетворительно"
        elif score < 0.85:
            return "Хорошо"
        else:
            return "Отлично"
    
    def _generate_recommendations(
        self,
        component_details: Dict,
        is_ymyl: bool,
        industry: str,
        overall_score: float
    ) -> List[str]:
        """Генерация комплексных рекомендаций на основе всех компонентов анализа"""
        recommendations = []
        
        # Базовый порог для рекомендаций, для YMYL более строгий
        threshold = 0.6 if is_ymyl else 0.4
        
        # Рекомендации по экспертизе
        expertise = component_details['expertise']
        if expertise['score'] < threshold:
            recommendations.extend([
                "Добавьте явную информацию о профессиональном опыте автора и его квалификации",
                "Включите подтверждение компетенции автора (образование, сертификаты, опыт работы)",
                "Добавьте экспертное мнение по ключевым вопросам темы"
            ])
            
            # Отраслевые рекомендации
            if industry == 'finance':
                recommendations.append("Укажите финансовую квалификацию автора (CFA, MBA, кандидат экономических наук)")
            elif industry == 'health':
                recommendations.append("Укажите медицинскую квалификацию и специализацию экспертов (врач, к.м.н.)")
            elif industry == 'legal':
                recommendations.append("Добавьте юридическую квалификацию и опыт работы в данной области права")
            elif industry == 'tech':
                recommendations.append("Укажите технический опыт и профессиональную специализацию автора")
                
        # Рекомендации по авторитетности
        authority = component_details['authority']
        if authority['score'] < threshold:
            recommendations.extend([
                "Добавьте ссылки на авторитетные источники по теме",
                "Включите актуальные статистические данные из проверенных исследований",
                "Цитируйте признанных экспертов отрасли"
            ])
            
            # Отраслевые рекомендации
            if industry == 'finance':
                recommendations.append("Цитируйте официальные финансовые учреждения и регуляторы (ЦБ, Минфин)")
            elif industry == 'health':
                recommendations.append("Ссылайтесь на рецензируемые медицинские публикации и рекомендации ВОЗ")
            elif industry == 'legal':
                recommendations.append("Добавьте ссылки на действующее законодательство и судебную практику")
            elif industry == 'tech':
                recommendations.append("Ссылайтесь на официальную документацию и технические спецификации")
        
        # Рекомендации по доверию
        trust = component_details['trust']
        if trust['score'] < threshold:
            recommendations.extend([
                "Добавьте даты публикации и обновления материала",
                "Укажите методологию и источники используемых данных",
                "Добавьте раздел с раскрытием информации о возможных конфликтах интересов"
            ])
            
            # Отраслевые рекомендации для доверия
            if industry == 'finance':
                recommendations.append("Добавьте дисклеймер о финансовых рисках и правовом статусе информации")
            elif industry == 'health':
                recommendations.append("Укажите предупреждения о противопоказаниях и необходимости консультации с врачом")
            elif industry == 'legal':
                recommendations.append("Добавьте отказ от ответственности и уточнение, что материал не является юридической консультацией")
        
        # Рекомендации по структуре
        structure = component_details['structure']
        if structure['score'] < threshold:
            recommendations.extend([
                "Улучшите структуру контента, добавив более четкие разделы и подзаголовки",
                "Включите списки и таблицы для лучшего структурирования информации",
                f"Добавьте цитаты экспертов в формате прямой речи (найдено только {structure['elements'].get('citations', 0)})"
            ])
        
        # Рекомендации по семантике
        semantics = component_details['semantics']
        if semantics['coherence_score'] < threshold:
            recommendations.extend([
                "Усильте тематическую связность текста, более последовательно развивая ключевые идеи",
                "Используйте специализированную терминологию, характерную для данной темы"
            ])
        
        # Рекомендации по цитированию
        citations = component_details['citations']
        if citations['score'] < threshold:
            recommendations.extend([
                "Добавьте больше прямых цитат от экспертов и специалистов отрасли",
                "Включите мнения разных экспертов для показа разных точек зрения"
            ])
        
        # Рекомендации по внешним ссылкам
        external_links = component_details['external_links']
        if external_links['score'] < threshold and 'details' in external_links:
            link_details = external_links['details']
            if link_details.get('total_links', 0) < 3:
                recommendations.append("Добавьте больше ссылок на авторитетные внешние источники")
            elif link_details.get('authority_ratio', 0) < 0.5:
                recommendations.append("Используйте более авторитетные источники для ссылок (академические, правительственные)")
        
        # Специальные рекомендации для YMYL сайтов
        if is_ymyl:
            if overall_score < 0.7:
                recommendations.append("КРИТИЧНО ДЛЯ YMYL: добавьте раздел 'Об авторе' с подробной информацией о квалификации")
                recommendations.append("КРИТИЧНО ДЛЯ YMYL: укажите дату последнего обновления информации и процесс проверки фактов")
                recommendations.append("КРИТИЧНО ДЛЯ YMYL: проверьте все утверждения на актуальность и подтвердите ссылками на авторитетные источники")
        
        # Приоритизация и дедупликация рекомендаций
        unique_recommendations = []
        seen = set()
        
        for rec in recommendations:
            # Упрощенный хеш на основе первых слов
            rec_key = ' '.join(rec.split()[:3]).lower()
            if rec_key not in seen:
                seen.add(rec_key)
                unique_recommendations.append(rec)
        
        # Ограничиваем количество рекомендаций для удобства
        return unique_recommendations[:12]  # Максимум 12 рекомендаций
    
    def train_model(
        self, 
        training_data: List[Dict[str, Union[Dict[str, float], float]]],
        output_path: str = 'eeat_model.joblib'
    ) -> None:
        """
        Обучение модели машинного обучения для оценки E-E-A-T
        
        Args:
            training_data: Список словарей с компонентами E-E-A-T и итоговыми оценками
            output_path: Путь для сохранения модели
        """
        try:
            # Импортируем библиотеки только при использовании функции
            try:
                from sklearn.ensemble import RandomForestRegressor
                import joblib
                import numpy as np
            except ImportError:
                logger.error("sklearn или joblib не установлены. Обучение модели недоступно.")
                return
                
            # Подготовка данных для обучения
            X = []
            y = []
            
            for item in training_data:
                features = [
                    item.get('expertise_score', 0.0),
                    item.get('authority_score', 0.0),
                    item.get('trust_score', 0.0),
                    item.get('structural_score', 0.0),
                    item.get('semantic_coherence_score', 0.0),
                    item.get('citation_score', 0.0),
                    item.get('external_links_score', 0.0),
                    1 if item.get('ymyl_status', False) else 0
                ]
                X.append(features)
                y.append(item.get('overall_eeat_score', 0.0))
            
            # Проверка наличия данных
            if not X or not y:
                logger.error("Нет данных для обучения модели.")
                return
            
            # Инициализация и обучение модели
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=5,
                random_state=42
            )
            model.fit(X, y)
            
            # Оценка качества модели
            y_pred = model.predict(X)
            mse = np.mean((np.array(y) - y_pred) ** 2)
            logger.info(f"Модель обучена. MSE на тренировочных данных: {mse:.4f}")
            
            # Сохранение модели
            joblib.dump(model, output_path)
            self.model = model
            self.model_path = output_path
            logger.info(f"Модель сохранена в {output_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Получение важности признаков из модели"""
        if not self.model:
            return {}
            
        try:
            feature_names = [
                'expertise_score',
                'authority_score',
                'trust_score',
                'structural_score',
                'semantic_coherence_score',
                'citation_score',
                'external_links_score',
                'ymyl_status'
            ]
            
            importance = self.model.feature_importances_
            return {name: float(imp) for name, imp in zip(feature_names, importance)}
            
        except Exception as e:
            logger.error(f"Ошибка при получении важности признаков: {str(e)}")
            return {}
    
    def generate_eeat_report(
        self, 
        analysis_result: Dict[str, Union[float, Dict, List]]
    ) -> str:
        """
        Генерация подробного отчета по E-E-A-T в формате Markdown
        
        Args:
            analysis_result: Результат анализа E-E-A-T
            
        Returns:
            Отчет в формате Markdown
        """
        industry = analysis_result.get('industry', 'не указана')
        ymyl_status = "Да" if analysis_result.get('ymyl_status', False) else "Нет"
        overall_score = analysis_result.get('overall_eeat_score', 0.0)
        expertise_score = analysis_result.get('expertise_score', 0.0)
        authority_score = analysis_result.get('authority_score', 0.0)
        trust_score = analysis_result.get('trust_score', 0.0)
        
        report = [
            "# Отчет по анализу E-E-A-T\n",
            f"**Отрасль:** {industry}",
            f"**YMYL-тематика:** {ymyl_status}",
            f"**Общая оценка E-E-A-T:** {overall_score:.2f} ({self._get_status(overall_score)})\n",
            "## Основные компоненты\n",
            f"| Компонент | Оценка | Статус |",
            f"|-----------|--------|--------|",
            f"| Экспертиза (E) | {expertise_score:.2f} | {self._get_status(expertise_score)} |",
            f"| Авторитетность (A) | {authority_score:.2f} | {self._get_status(authority_score)} |",
            f"| Доверие (T) | {trust_score:.2f} | {self._get_status(trust_score)} |",
            f"| Структура | {analysis_result.get('structural_score', 0.0):.2f} | {self._get_status(analysis_result.get('structural_score', 0.0))} |",
            f"| Семантическая связность | {analysis_result.get('semantic_coherence_score', 0.0):.2f} | {self._get_status(analysis_result.get('semantic_coherence_score', 0.0))} |"
        ]
        
        # Добавление рекомендаций
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            report.append("\n## Рекомендации по улучшению\n")
            for rec in recommendations:
                report.append(f"- {rec}")
        
        # Детали по найденным маркерам
        component_details = analysis_result.get('component_details', {})
        
        if 'expertise' in component_details:
            expertise = component_details['expertise']
            found_markers = expertise.get('found_markers', [])
            if found_markers:
                report.append("\n## Найденные маркеры экспертизы\n")
                for marker in found_markers[:10]:  # Ограничиваем для краткости
                    report.append(f"- {marker}")
                if len(found_markers) > 10:
                    report.append(f"- ... и еще {len(found_markers) - 10}")
        
        # Информация о цитатах
        if 'citations' in component_details:
            citations = component_details['citations']
            examples = citations.get('examples', [])
            if examples:
                report.append("\n## Примеры цитат\n")
                for example in examples:
                    report.append(f"- \"{example}\"")
        
        # Внешние ссылки
        if 'external_links' in component_details:
            external_links = component_details['external_links']
            if 'details' in external_links:
                link_details = external_links['details']
                if link_details:
                    report.append("\n## Анализ внешних ссылок\n")
                    report.append(f"Всего ссылок: {link_details.get('total_links', 0)}")
                    report.append(f"Авторитетных источников: {link_details.get('authority_links', 0)}")
                    
                    if 'link_details' in link_details:
                        links = link_details['link_details']
                        report.append("\n| Домен | Авторитетный | Заголовок |")
                        report.append("|-------|--------------|-----------|")
                        for url, details in list(links.items())[:5]:  # Ограничиваем для краткости
                            domain = details.get('domain', 'Н/Д')
                            is_auth = "Да" if details.get('is_authority', False) else "Нет"
                            title = details.get('title', 'Н/Д')
                            if len(title) > 30:
                                title = title[:27] + "..."
                            report.append(f"| {domain} | {is_auth} | {title} |")
        
        return "\n".join(report)


def generate_synthetic_training_data(count: int = 100) -> List[Dict[str, Union[float, bool, str]]]:
    """
    Генерация синтетических данных для обучения модели E-E-A-T
    
    Args:
        count: Количество примеров для генерации
        
    Returns:
        Список словарей с данными
    """
    import random
    
    data = []
    industries = ['finance', 'health', 'legal', 'tech', 'ecommerce', 'default']
    
    for _ in range(count):
        # Случайная отрасль
        industry = random.choice(industries)
        
        # YMYL статус (финансы, здоровье и юридическая тематика всегда YMYL)
        is_ymyl = industry in ['finance', 'health', 'legal']
        
        # Генерация базовых компонентов
        expertise_score = random.uniform(0.1, 0.95)
        authority_score = random.uniform(0.1, 0.95)
        trust_score = random.uniform(0.1, 0.95)
        structural_score = random.uniform(0.1, 0.95)
        semantic_score = random.uniform(0.1, 0.95)
        citation_score = random.uniform(0.1, 0.95)
        external_links_score = random.uniform(0.1, 0.95)
        
        # Для YMYL отраслей, общая оценка сильнее зависит от доверия и авторитетности
        if is_ymyl:
            overall_score = (
                expertise_score * 0.20 +
                authority_score * 0.25 +
                trust_score * 0.30 +
                structural_score * 0.05 +
                semantic_score * 0.10 +
                citation_score * 0.05 +
                external_links_score * 0.05
            )
        else:
            overall_score = (
                expertise_score * 0.25 +
                authority_score * 0.20 +
                trust_score * 0.20 +
                structural_score * 0.10 +
                semantic_score * 0.15 +
                citation_score * 0.05 +
                external_links_score * 0.05
            )
        
        # Добавляем небольшой случайный шум для реалистичности
        overall_score = min(1.0, max(0.0, overall_score + random.uniform(-0.05, 0.05)))
        
        data.append({
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'semantic_coherence_score': semantic_score,
            'citation_score': citation_score,
            'external_links_score': external_links_score,
            'overall_eeat_score': overall_score,
            'ymyl_status': is_ymyl,
            'industry': industry
        })
    
    return data
