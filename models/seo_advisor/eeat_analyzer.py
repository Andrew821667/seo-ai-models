from typing import Dict, List, Optional, Union, Any
import re
from collections import Counter

class EEATAnalyzer:
    """Анализатор E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)"""
    
    def __init__(self):
        # Маркеры опыта и экспертизы
        self.expertise_markers = [
            'опыт', 'эксперт', 'специалист', 'профессионал', 'квалификация',
            'сертифицированный', 'компетентный', 'практик', 'исследователь', 'аналитик',
            'образование', 'обучение', 'диплом', 'степень', 'награда', 'стаж', 'практика',
            'достижение', 'сертификат', 'знания', 'навык', 'умение', 'подготовка'
        ]
        
        # Маркеры авторитетности
        self.authority_markers = [
            'исследование', 'статистика', 'данные', 'доказано', 'согласно', 
            'по мнению экспертов', 'научно', 'источник', 'цитата', 'ссылка',
            'авторитет', 'эксперты рекомендуют', 'в соответствии с', 'рецензируемый',
            'признанный', 'известный', 'уважаемый', 'влиятельный', 'популярный',
            'научная статья', 'журнал', 'публикация', 'библиография'
        ]
        
        # Маркеры доверия - РАСШИРЕНО с учетом дат, методологий и раскрытия информации
        self.trust_markers = [
            'достоверный', 'проверенный', 'надежный', 'точный', 'подтвержденный',
            'официальный', 'гарантированный', 'безопасный', 'проверка фактов', 'прозрачность',
            'методология', 'метод', 'данные показывают', 'доказательство', 'подтверждено',
            'публикация', 'обновлено', 'раскрытие информации', 'отказ от ответственности', 'дисклеймер',
            'источники данных', 'обновление', 'актуальность', 'текущий', 'сравнение',
            'анализ', 'критерии', 'протокол', 'этика', 'независимый', 'объективный',
            'рецензирование', 'верификация', 'аудит', 'сертификация', 'стандарт'
        ]
        
        # Добавляем шаблоны регулярных выражений для поиска особых маркеров доверия
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
            r'(?:рецензирован|проверен)[^.!?]*эксперт'  # Рецензирование экспертами
        ]
        
        # Специфичные для отраслей маркеры
        self.industry_markers = {
            'finance': {
                'expertise': ['финансовый аналитик', 'экономист', 'банкир', 'финансовый советник', 'аудитор'],
                'authority': ['центральный банк', 'минфин', 'регулятор', 'налоговая служба', 'финансовый отчет'],
                'trust': ['лицензия', 'сертификат', 'регулируемый', 'финансовая отчетность', 'документация']
            },
            'health': {
                'expertise': ['врач', 'медицинский', 'фармацевт', 'терапевт', 'хирург', 'клиника', 'больница'],
                'authority': ['министерство здравоохранения', 'воз', 'медицинский журнал', 'клинические испытания'],
                'trust': ['пациент', 'диагноз', 'лечение', 'протокол лечения', 'медицинская этика', 'показания']
            },
            'legal': {
                'expertise': ['юрист', 'адвокат', 'нотариус', 'прокурор', 'судья', 'законодательство', 'кодекс'],
                'authority': ['суд', 'кодекс', 'закон', 'правовой акт', 'постановление', 'решение суда'],
                'trust': ['правовая норма', 'законодательный акт', 'нормативный', 'юридическая сила', 'легальный']
            },
            'tech': {
                'expertise': ['разработчик', 'программист', 'инженер', 'технолог', 'айти-специалист', 'дизайнер'],
                'authority': ['техническая документация', 'спецификация', 'стандарт', 'api', 'протокол'],
                'trust': ['версия', 'релиз', 'патч', 'обновление', 'баг-фикс', 'тестирование', 'валидация']
            }
        }
        
        # YMYL отрасли
        self.ymyl_industries = {
            'finance': True,
            'health': True,
            'legal': True,
            'insurance': True,
            'medical': True,
            'crypto': True,
            'investment': True
        }
        
    def analyze(self, text: str, industry: str = 'default') -> Dict[str, Union[float, Dict, List]]:
        """
        Оценка E-E-A-T сигналов в тексте
        
        Args:
            text: Анализируемый текст
            industry: Отрасль контента
            
        Returns:
            Словарь с оценками и рекомендациями
        """
        text_lower = text.lower()
        is_ymyl = self.ymyl_industries.get(industry, False)
        
        # Получение маркеров, специфичных для отрасли
        industry_specific_markers = self.industry_markers.get(industry, {})
        
        # Оценка опыта и экспертизы с учетом отрасли
        expertise_markers = self.expertise_markers[:]
        if 'expertise' in industry_specific_markers:
            expertise_markers.extend(industry_specific_markers['expertise'])
        expertise_score = self._evaluate_markers(text_lower, expertise_markers)
        
        # Оценка авторитетности с учетом отрасли
        authority_markers = self.authority_markers[:]
        if 'authority' in industry_specific_markers:
            authority_markers.extend(industry_specific_markers['authority'])
        authority_score = self._evaluate_markers(text_lower, authority_markers)
        
        # Оценка доверия с учетом отрасли
        trust_markers = self.trust_markers[:]
        if 'trust' in industry_specific_markers:
            trust_markers.extend(industry_specific_markers['trust'])
        trust_score = self._evaluate_trust(text, text_lower, trust_markers)
        
        # Оценка структурных элементов, укрепляющих E-E-A-T
        structural_score = self._evaluate_structure(text)
        
        # Средневзвешенная оценка - УЛУЧШЕНО с акцентом на доверие для YMYL
        # Для YMYL отраслей увеличиваем вес доверия и авторитетности
        if is_ymyl:
            overall_eeat_score = (
                expertise_score * 0.25 + 
                authority_score * 0.35 + 
                trust_score * 0.30 +  # Увеличенный вес доверия для YMYL
                structural_score * 0.10
            )
        else:
            overall_eeat_score = (
                expertise_score * 0.30 + 
                authority_score * 0.30 + 
                trust_score * 0.25 +
                structural_score * 0.15
            )
        
        # Подробный анализ компонентов
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
            'structure': {
                'score': structural_score,
                'elements': self._count_structural_elements(text),
                'status': self._get_status(structural_score)
            }
        }
        
        # Формирование рекомендаций с учетом YMYL и отрасли
        recommendations = self._generate_recommendations(
            expertise_score, authority_score, trust_score, structural_score,
            is_ymyl, industry, component_details
        )
        
        return {
            'expertise_score': expertise_score,
            'authority_score': authority_score,
            'trust_score': trust_score,
            'structural_score': structural_score,
            'overall_eeat_score': overall_eeat_score,
            'recommendations': recommendations,
            'component_details': component_details,
            'ymyl_status': is_ymyl,
            'industry': industry
        }
    
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
        """УЛУЧШЕННАЯ оценка доверия с учетом дат, методологий и других специальных сигналов"""
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
    
    def _evaluate_structure(self, text: str) -> float:
        """Оценка структурных элементов, укрепляющих E-E-A-T"""
        score = 0.0
        
        # Проверка наличия цитат
        quotes = text.count('"') + text.count("'") + text.count("«") + text.count("»")
        quotes_score = min(quotes / 8, 1.0)  # Учитываем открывающие и закрывающие кавычки
        
        # Проверка наличия ссылок на источники
        sources_ref = (
            text.lower().count('источник') + 
            text.lower().count('ссылк') + 
            text.lower().count('http') + 
            text.lower().count('www')
        )
        sources_score = min(sources_ref / 4, 1.0)
        
        # Проверка наличия структурированных данных (списки)
        lists = text.count('-') + text.count('*') + text.count('•') + text.count('1.')
        lists_score = min(lists / 8, 1.0)
        
        # Проверка наличия подзаголовков
        headers = text.count('#') + text.count('##') + text.count('###')
        headers_score = min(headers / 5, 1.0)
        
        # Проверка наличия таблиц
        tables = text.count('|') / 10  # В таблицах обычно много вертикальных черт
        tables_score = min(tables, 1.0)
        
        # Взвешенная оценка структуры
        score = (
            quotes_score * 0.2 +
            sources_score * 0.3 +
            lists_score * 0.2 +
            headers_score * 0.2 +
            tables_score * 0.1
        )
        
        return score
    
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
    
    def _count_structural_elements(self, text: str) -> Dict[str, int]:
        """Подсчет структурных элементов в тексте"""
        return {
            'quotes': text.count('"') + text.count("'") + text.count("«") + text.count("»"),
            'lists': text.count('-') + text.count('*') + text.count('•') + text.count('1.'),
            'headers': text.count('#') + text.count('##') + text.count('###'),
            'tables': text.count('|') // 10,
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()])
        }
    
    def _get_status(self, score: float) -> str:
        """Получение текстового статуса на основе оценки"""
        if score < 0.3:
            return "Требует улучшения"
        elif score < 0.6:
            return "Удовлетворительно"
        elif score < 0.8:
            return "Хорошо"
        else:
            return "Отлично"
    
    def _generate_recommendations(
        self, 
        expertise_score: float, 
        authority_score: float, 
        trust_score: float,
        structural_score: float,
        is_ymyl: bool,
        industry: str,
        component_details: Dict
    ) -> List[str]:
        """Генерация рекомендаций на основе оценок E-E-A-T с учетом отрасли и YMYL"""
        recommendations = []
        
        # Базовый порог для рекомендаций, для YMYL более строгий
        threshold = 0.5 if is_ymyl else 0.3
        
        # Рекомендации по экспертизе
        if expertise_score < threshold:
            recommendations.extend([
                "Добавьте информацию о профессиональном опыте автора в данной области",
                "Включите подтверждение компетенции автора (образование, сертификаты, опыт работы)",
                "Добавьте экспертное мнение по ключевым вопросам темы"
            ])
            
            # Отраслевые рекомендации
            if industry == 'finance':
                recommendations.append("Укажите финансовую квалификацию автора или консультантов")
            elif industry == 'health':
                recommendations.append("Укажите медицинскую квалификацию и специализацию экспертов")
            elif industry == 'legal':
                recommendations.append("Добавьте юридическую квалификацию и опыт работы в данной области права")
            elif industry == 'tech':
                recommendations.append("Укажите технический опыт и профессиональную специализацию автора")
                
        elif expertise_score < 0.7:
            recommendations.append("Усильте демонстрацию экспертности, добавив примеры из практики")
            
            if is_ymyl:
                recommendations.append("Для YMYL темы: конкретизируйте профессиональную квалификацию и опыт")
        
        # Рекомендации по авторитетности
        if authority_score < threshold:
            recommendations.extend([
                "Добавьте ссылки на авторитетные источники по теме",
                "Включите статистические данные из проверенных исследований",
                "Цитируйте признанных экспертов отрасли"
            ])
            
            # Отраслевые рекомендации
            if industry == 'finance':
                recommendations.append("Цитируйте официальные финансовые учреждения и регуляторы")
            elif industry == 'health':
                recommendations.append("Ссылайтесь на рецензируемые медицинские исследования и рекомендации ВОЗ")
            elif industry == 'legal':
                recommendations.append("Добавьте ссылки на действующее законодательство и судебную практику")
            elif industry == 'tech':
                recommendations.append("Ссылайтесь на технические спецификации и официальную документацию")
                
        elif authority_score < 0.7:
            recommendations.append("Усильте авторитетность контента, добавив больше ссылок на внешние источники")
            
            if is_ymyl:
                recommendations.append("Для YMYL темы: подкрепите каждое значимое утверждение ссылкой на авторитетный источник")
        
        # Рекомендации по доверию
        if trust_score < threshold:
            recommendations.extend([
                "Добавьте даты публикации и обновления материала",
                "Включите методологию или источники используемых данных",
                "Добавьте раздел с раскрытием информации о возможных конфликтах интересов",
                "Включите точные цифры и статистику с указанием источников"
            ])
            
            # Отраслевые рекомендации
            if industry == 'finance':
                recommendations.append("Добавьте дисклеймер о рисках и правовой статус финансовой информации")
            elif industry == 'health':
                recommendations.append("Укажите предупреждения о противопоказаниях и необходимости консультации с врачом")
            elif industry == 'legal':
                recommendations.append("Добавьте отказ от ответственности и уточнение о том, что материал не является юридической консультацией")
            elif industry == 'tech':
                recommendations.append("Укажите версии продуктов и платформ, для которых актуальна информация")
                
        elif trust_score < 0.7:
            recommendations.extend([
                "Расширьте информацию о методологии сбора данных",
                "Добавьте больше точных дат и временных рамок",
                "Усильте прозрачность с помощью раскрытия дополнительной информации"
            ])
            
            if is_ymyl:
                recommendations.append("Для YMYL темы: добавьте информацию о регулярном обновлении и проверке контента")
        
        # Рекомендации по структуре
        if structural_score < threshold:
            recommendations.extend([
                "Улучшите структуру контента, добавив подзаголовки и разделы",
                "Включите списки и таблицы для структурирования информации",
                "Добавьте цитаты экспертов в формате прямой речи"
            ])
        elif structural_score < 0.7:
            recommendations.append("Улучшите структуру текста для более наглядной демонстрации экспертности")
        
        # Специальные рекомендации для YMYL сайтов
        if is_ymyl:
            overall_score = (expertise_score + authority_score + trust_score + structural_score) / 4
            if overall_score < 0.6:
                recommendations.append("ВАЖНО ДЛЯ YMYL: добавьте раздел 'Об авторе' с подробной информацией о квалификации")
                recommendations.append("ВАЖНО ДЛЯ YMYL: укажите дату последнего обновления информации")
                recommendations.append("ВАЖНО ДЛЯ YMYL: проверьте все факты и данные на актуальность и достоверность")
        
        # Ограничиваем количество рекомендаций для удобства
        if len(recommendations) > 10:
            recommendations = recommendations[:10]
        
        return recommendations
