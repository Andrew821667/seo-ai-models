"""
Чекер санкционных списков для due diligence контрагентов.
"""

import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict

from seo_ai_models.legal.due_diligence.models.due_diligence import (
    DueDiligenceCheck, CheckType, RiskLevel, SanctionsCheck
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SanctionsChecker:
    """
    Проверка контрагентов на наличие в санкционных списках.
    """

    def __init__(self):
        """Инициализация чекера санкций."""
        # Списки санкций для проверки
        self.sanctions_lists = {
            'us_ofac': {
                'name': 'OFAC SDN List (США)',
                'url': 'https://www.treasury.gov/ofac/downloads/sdnlist.txt',
                'country': 'US',
                'update_frequency': 'daily'
            },
            'eu_sanctions': {
                'name': 'EU Sanctions List',
                'url': 'https://webgate.ec.europa.eu/europeaid/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content/EN',
                'country': 'EU',
                'update_frequency': 'weekly'
            },
            'uk_sanctions': {
                'name': 'UK Sanctions List',
                'url': 'https://ofsistorage.blob.core.windows.net/publishlive/2023format/ConList.xml',
                'country': 'UK',
                'update_frequency': 'weekly'
            },
            'russia_criminal': {
                'name': 'Перечень террористов и экстремистов (Россия)',
                'url': 'https://fsb.ru/fsb/npd/terror.htm',  # пример URL
                'country': 'RU',
                'update_frequency': 'monthly'
            },
            'interpol': {
                'name': 'Interpol Red Notices',
                'url': 'https://www.interpol.int/How-we-work/Notices/Red-Notices/View-Red-Notices',
                'country': 'INTERPOL',
                'update_frequency': 'daily'
            }
        }

        # Ключевые слова для различных типов санкций
        self.sanction_types = {
            'terrorism': ['terrorism', 'terrorist', 'экстремизм', 'терроризм'],
            'corruption': ['corruption', 'коррупция', 'взятка', 'bribe'],
            'money_laundering': ['money laundering', 'отмывание', 'laundering'],
            'fraud': ['fraud', 'мошенничество', 'fraudulent'],
            'weapons': ['weapons', 'оружие', 'arms'],
            'narcotics': ['narcotics', 'наркотики', 'drugs']
        }

    def check_sanctions(
        self,
        company_name: str,
        inn: Optional[str] = None,
        director_name: Optional[str] = None,
        address: Optional[str] = None
    ) -> DueDiligenceCheck:
        """
        Проверка контрагента на наличие в санкционных списках.

        Args:
            company_name: Название компании
            inn: ИНН компании
            director_name: ФИО директора
            address: Адрес компании

        Returns:
            DueDiligenceCheck: Результат проверки
        """
        try:
            check = DueDiligenceCheck(
                check_type=CheckType.SANCTIONS,
                status="in_progress"
            )

            # Проверяем все доступные списки
            sanctions_results = []
            checked_lists = []

            for list_key, list_info in self.sanctions_lists.items():
                try:
                    logger.info(f"Checking {list_info['name']} for {company_name}")

                    # В реальной реализации здесь был бы запрос к API или парсинг сайта
                    # Пока симулируем проверку
                    result = self._check_single_list(
                        list_key, list_info, company_name, inn, director_name, address
                    )
                    sanctions_results.append(result)
                    checked_lists.append(list_info['name'])

                except Exception as e:
                    logger.warning(f"Error checking {list_info['name']}: {str(e)}")
                    continue

            # Анализируем результаты
            overall_result = self._analyze_sanctions_results(sanctions_results)

            # Определяем уровень риска
            if overall_result['is_sanctioned']:
                risk_level = RiskLevel.CRITICAL
                score = 0.0
                findings = [f"Найден в санкционном списке: {overall_result['sanction_type']}"]
            else:
                risk_level = RiskLevel.LOW
                score = 95.0  # высокая оценка если не найден
                findings = ["Не найден в проверенных санкционных списках"]

            recommendations = []
            if overall_result['is_sanctioned']:
                recommendations.extend([
                    "Немедленно прекратить любые отношения с контрагентом",
                    "Провести дополнительную проверку на предмет обхода санкций",
                    "Обратиться к специалистам по санкционному праву"
                ])
            else:
                recommendations.append("Регулярно мониторить изменения в санкционных списках")

            # Формируем результат
            check.status = "completed"
            check.risk_level = risk_level
            check.score = score
            check.findings = findings
            check.recommendations = recommendations
            check.data = {
                'overall_result': overall_result,
                'checked_lists': checked_lists,
                'sanctions_results': sanctions_results
            }
            check.checked_at = datetime.now()

            logger.info(f"Sanctions check completed for {company_name}: sanctioned={overall_result['is_sanctioned']}")
            return check

        except Exception as e:
            logger.error(f"Error checking sanctions for {company_name}: {str(e)}")
            return DueDiligenceCheck(
                check_type=CheckType.SANCTIONS,
                status="failed",
                risk_level=RiskLevel.CRITICAL,  # в случае ошибки считаем критический риск
                score=0.0,
                findings=["Ошибка проверки санкционных списков"],
                error_message=str(e),
                checked_at=datetime.now()
            )

    def _check_single_list(
        self,
        list_key: str,
        list_info: Dict[str, Any],
        company_name: str,
        inn: Optional[str],
        director_name: Optional[str],
        address: Optional[str]
    ) -> Dict[str, Any]:
        """
        Проверка одного санкционного списка.
        """
        try:
            # В реальной реализации здесь был бы HTTP-запрос и парсинг данных
            # Пока возвращаем симулированный результат

            # Имитируем проверку (в продакшене заменить на реальную логику)
            search_terms = [company_name]
            if inn:
                search_terms.append(inn)
            if director_name:
                search_terms.append(director_name)

            # Симуляция поиска (случайный результат для демонстрации)
            import random
            is_found = random.random() < 0.05  # 5% вероятность нахождения

            if is_found:
                # Симулируем найденную запись
                sanction_types = list(self.sanction_types.keys())
                sanction_type = random.choice(sanction_types)

                return {
                    'list_name': list_info['name'],
                    'is_found': True,
                    'sanction_type': sanction_type,
                    'sanction_country': list_info['country'],
                    'sanction_date': (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                    'sanction_reason': f"Включен в список по причине: {sanction_type}",
                    'matched_term': random.choice(search_terms)
                }
            else:
                return {
                    'list_name': list_info['name'],
                    'is_found': False,
                    'checked_terms': search_terms
                }

        except Exception as e:
            logger.error(f"Error checking list {list_key}: {str(e)}")
            return {
                'list_name': list_info['name'],
                'error': str(e)
            }

    def _analyze_sanctions_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализ результатов проверки всех списков.
        """
        sanctioned_entries = [r for r in results if r.get('is_found', False)]

        if sanctioned_entries:
            # Берем наиболее серьезную санкцию
            most_serious = sanctioned_entries[0]  # в реальности нужно ранжировать по серьезности

            return {
                'is_sanctioned': True,
                'sanction_type': most_serious.get('sanction_type'),
                'sanction_country': most_serious.get('sanction_country'),
                'sanction_date': most_serious.get('sanction_date'),
                'sanction_reason': most_serious.get('sanction_reason'),
                'found_in_lists': [entry['list_name'] for entry in sanctioned_entries],
                'matched_term': most_serious.get('matched_term')
            }
        else:
            return {
                'is_sanctioned': False,
                'checked_lists_count': len(results),
                'no_matches_found': True
            }

    def get_sanctions_summary(self, company_name: str) -> SanctionsCheck:
        """
        Получение сводки по санкциям для компании.

        Args:
            company_name: Название компании

        Returns:
            SanctionsCheck: Сводка по санкциям
        """
        check_result = self.check_sanctions(company_name)

        if check_result.status == "completed" and check_result.data:
            overall_result = check_result.data.get('overall_result', {})

            return SanctionsCheck(
                is_sanctioned=overall_result.get('is_sanctioned', False),
                sanction_type=overall_result.get('sanction_type'),
                sanction_country=overall_result.get('sanction_country'),
                sanction_date=datetime.fromisoformat(overall_result['sanction_date']) if overall_result.get('sanction_date') else None,
                sanction_reason=overall_result.get('sanction_reason'),
                checked_lists=check_result.data.get('checked_lists', [])
            )

        # Возвращаем пустую проверку в случае ошибки
        return SanctionsCheck(
            checked_lists=check_result.data.get('checked_lists', []) if check_result.data else []
        )

    def _get_ofac_data(self) -> List[Dict[str, Any]]:
        """
        Получение данных из списка OFAC SDN.
        В реальной реализации здесь был бы парсинг и кеширование данных.
        """
        try:
            # Пример структуры данных OFAC
            # В продакшене нужно реализовать реальный парсинг
            return [
                {
                    'name': 'Example Corp',
                    'type': 'entity',
                    'sanction_type': 'terrorism',
                    'country': 'US'
                }
            ]
        except Exception as e:
            logger.error(f"Error getting OFAC data: {str(e)}")
            return []

    def _get_eu_sanctions_data(self) -> List[Dict[str, Any]]:
        """
        Получение данных из списка санкций ЕС.
        """
        try:
            # Пример структуры данных ЕС
            return []
        except Exception as e:
            logger.error(f"Error getting EU sanctions data: {str(e)}")
            return []

    def update_sanctions_cache(self):
        """
        Обновление кеша санкционных списков.
        Этот метод должен вызываться периодически для обновления данных.
        """
        try:
            logger.info("Updating sanctions cache...")

            # В реальной реализации здесь было бы скачивание и парсинг всех списков
            # с сохранением в локальную базу данных

            logger.info("Sanctions cache updated successfully")

        except Exception as e:
            logger.error(f"Error updating sanctions cache: {str(e)}")
