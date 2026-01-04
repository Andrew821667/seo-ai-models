"""
Парсер данных арбитражных судов РФ с сайта my.arbitr.ru
Адаптирован на основе унифицированного парсера SEO AI Models
"""

import logging
import time
import re
import json
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse, urljoin, parse_qs
from datetime import datetime
from dataclasses import asdict

import requests
from bs4 import BeautifulSoup

from seo_ai_models.legal.court_practice.models.court_case import (
    CourtCase, CourtDecision, Party, CourtType, CaseStatus, CaseCategory
)
from seo_ai_models.common.utils.enhanced_text_processor import EnhancedTextProcessor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ArbitrageCourtParser:
    """
    Парсер данных арбитражных судов РФ.
    Извлекает информацию о судебных делах с сайта my.arbitr.ru
    """

    BASE_URL = "https://my.arbitr.ru"
    SEARCH_URL = "https://my.arbitr.ru/#/cases"

    def __init__(
        self,
        user_agent: str = "LegalAISystem ArbitrageParser/1.0",
        delay: float = 2.0,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Инициализация парсера арбитражных судов.

        Args:
            user_agent: User-Agent для запросов
            delay: Задержка между запросами в секундах
            max_retries: Максимальное количество повторных попыток
            timeout: Таймаут запроса в секундах
        """
        self.user_agent = user_agent
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout

        # Инициализация сессии
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # Инициализация текстового процессора
        self.text_processor = EnhancedTextProcessor()

        logger.info("ArbitrageCourtParser initialized")

    def search_cases(
        self,
        query: str = "",
        court_region: Optional[str] = None,
        case_category: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        max_results: int = 50
    ) -> List[CourtCase]:
        """
        Поиск судебных дел по заданным критериям.

        Args:
            query: Поисковый запрос (наименование сторон, номер дела и т.д.)
            court_region: Регион суда
            case_category: Категория дела
            date_from: Дата начала периода
            date_to: Дата окончания периода
            max_results: Максимальное количество результатов

        Returns:
            List[CourtCase]: Список найденных судебных дел
        """
        cases = []

        try:
            # Формируем параметры поиска
            search_params = self._build_search_params(
                query=query,
                court_region=court_region,
                case_category=case_category,
                date_from=date_from,
                date_to=date_to
            )

            # Выполняем поиск
            search_url = f"{self.BASE_URL}/cases"
            logger.info(f"Searching cases with params: {search_params}")

            response = self._make_request(search_url, params=search_params)
            if not response:
                return cases

            # Парсим результаты поиска
            cases_data = self._parse_search_results(response.text)

            # Обрабатываем каждое дело
            for case_data in cases_data[:max_results]:
                try:
                    case = self._parse_case_details(case_data)
                    if case:
                        cases.append(case)

                    # Задержка между запросами
                    time.sleep(self.delay)

                except Exception as e:
                    logger.error(f"Error parsing case {case_data.get('caseNumber', 'unknown')}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error during case search: {str(e)}")

        logger.info(f"Found {len(cases)} cases")
        return cases

    def get_case_details(self, case_number: str) -> Optional[CourtCase]:
        """
        Получение детальной информации о судебном деле по номеру.

        Args:
            case_number: Номер судебного дела

        Returns:
            Optional[CourtCase]: Детали судебного дела или None
        """
        try:
            # Формируем URL для просмотра дела
            case_url = f"{self.BASE_URL}/CaseCourt/case/{case_number}/details"

            logger.info(f"Fetching case details for {case_number}")

            response = self._make_request(case_url)
            if not response:
                return None

            # Парсим детали дела
            return self._parse_case_page(response.text, case_number)

        except Exception as e:
            logger.error(f"Error fetching case details for {case_number}: {str(e)}")
            return None

    def _build_search_params(
        self,
        query: str = "",
        court_region: Optional[str] = None,
        case_category: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Формирование параметров поиска.
        """
        params = {}

        if query:
            params['search'] = query

        if court_region:
            params['courtRegion'] = court_region

        if case_category:
            params['caseCategory'] = case_category

        if date_from:
            params['dateFrom'] = date_from.strftime('%d.%m.%Y')

        if date_to:
            params['dateTo'] = date_to.strftime('%d.%m.%Y')

        return params

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = 'GET'
    ) -> Optional[requests.Response]:
        """
        Выполнение HTTP-запроса с повторными попытками.
        """
        for attempt in range(self.max_retries):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(
                        url,
                        params=params,
                        timeout=self.timeout
                    )
                else:
                    response = self.session.post(
                        url,
                        data=params,
                        timeout=self.timeout
                    )

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
                continue

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None

    def _parse_search_results(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Парсинг результатов поиска.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        cases_data = []

        # Ищем таблицу с результатами (адаптировать под структуру сайта)
        case_rows = soup.find_all('tr', class_=re.compile(r'case-row|case-item'))

        for row in case_rows:
            try:
                case_data = self._extract_case_from_row(row)
                if case_data:
                    cases_data.append(case_data)
            except Exception as e:
                logger.warning(f"Error parsing case row: {str(e)}")
                continue

        return cases_data

    def _extract_case_from_row(self, row) -> Optional[Dict[str, Any]]:
        """
        Извлечение данных дела из строки таблицы результатов поиска.
        """
        try:
            # Адаптировать под реальную структуру HTML сайта my.arbitr.ru
            cells = row.find_all('td')

            if len(cells) < 4:
                return None

            case_data = {
                'caseNumber': cells[0].get_text(strip=True),
                'courtName': cells[1].get_text(strip=True),
                'parties': cells[2].get_text(strip=True),
                'status': cells[3].get_text(strip=True),
                'url': row.find('a', href=True)['href'] if row.find('a', href=True) else None
            }

            return case_data

        except Exception as e:
            logger.warning(f"Error extracting case data from row: {str(e)}")
            return None

    def _parse_case_details(self, case_data: Dict[str, Any]) -> Optional[CourtCase]:
        """
        Парсинг деталей дела из кратких данных поиска.
        """
        case_number = case_data.get('caseNumber')
        if not case_number:
            return None

        return self.get_case_details(case_number)

    def _parse_case_page(self, html_content: str, case_number: str) -> Optional[CourtCase]:
        """
        Парсинг страницы с деталями судебного дела.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Извлекаем основные данные дела
            case_info = self._extract_case_info(soup, case_number)
            if not case_info:
                return None

            # Извлекаем участников
            parties = self._extract_parties(soup)

            # Извлекаем судебные решения
            decisions = self._extract_decisions(soup)

            # Извлекаем суть спора
            claim_info = self._extract_claim_info(soup)

            # Определяем категорию дела
            category = self._classify_case_category(
                case_info.get('description', ''),
                parties,
                claim_info
            )

            # Создаем объект дела
            case = CourtCase(
                case_number=case_number,
                court_type=CourtType.ARBITRAGE,
                court_name=case_info.get('court_name', ''),
                category=category,
                status=self._parse_case_status(case_info.get('status', '')),
                plaintiffs=[p for p in parties if p.role == 'plaintiff'],
                defendants=[p for p in parties if p.role == 'defendant'],
                third_parties=[p for p in parties if p.role == 'third_party'],
                claim_subject=claim_info.get('subject', ''),
                claim_amount=claim_info.get('amount'),
                description=case_info.get('description', ''),
                filing_date=case_info.get('filing_date'),
                decisions=decisions,
                source_url=f"{self.BASE_URL}/CaseCourt/case/{case_number}/details"
            )

            return case

        except Exception as e:
            logger.error(f"Error parsing case page {case_number}: {str(e)}")
            return None

    def _extract_case_info(self, soup: BeautifulSoup, case_number: str) -> Optional[Dict[str, Any]]:
        """
        Извлечение основной информации о деле.
        """
        try:
            case_info = {}

            # Ищем блок с информацией о деле
            case_block = soup.find('div', class_=re.compile(r'case-info|case-details'))
            if not case_block:
                # Альтернативный поиск
                case_block = soup.find('div', id=re.compile(r'caseInfo|caseDetails'))

            if case_block:
                # Извлекаем данные из блока
                case_info = self._parse_case_info_block(case_block)
            else:
                # Извлекаем данные из других элементов страницы
                case_info = self._parse_scattered_case_info(soup)

            case_info['case_number'] = case_number
            return case_info

        except Exception as e:
            logger.warning(f"Error extracting case info: {str(e)}")
            return None

    def _parse_case_info_block(self, block) -> Dict[str, Any]:
        """
        Парсинг блока с информацией о деле.
        """
        info = {}

        # Адаптировать под реальную структуру сайта
        # Примерная структура (нужно адаптировать под реальный HTML)

        # Суд
        court_elem = block.find('span', class_='court-name')
        info['court_name'] = court_elem.get_text(strip=True) if court_elem else ''

        # Статус
        status_elem = block.find('span', class_='case-status')
        info['status'] = status_elem.get_text(strip=True) if status_elem else ''

        # Дата подачи
        date_elem = block.find('span', class_='filing-date')
        if date_elem:
            try:
                info['filing_date'] = datetime.strptime(
                    date_elem.get_text(strip=True),
                    '%d.%m.%Y'
                )
            except ValueError:
                pass

        # Описание
        desc_elem = block.find('div', class_='case-description')
        info['description'] = desc_elem.get_text(strip=True) if desc_elem else ''

        return info

    def _parse_scattered_case_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Парсинг информации о деле из различных элементов страницы.
        """
        info = {}

        # Ищем элементы с информацией в разных частях страницы
        # Адаптировать под реальную структуру сайта my.arbitr.ru

        # Поиск по заголовкам и меткам
        labels = soup.find_all(['dt', 'label', 'strong'])

        for label in labels:
            text = label.get_text(strip=True).lower()

            if 'суд' in text or 'court' in text:
                next_elem = label.find_next(['dd', 'span', 'div'])
                if next_elem:
                    info['court_name'] = next_elem.get_text(strip=True)

            elif 'статус' in text or 'status' in text:
                next_elem = label.find_next(['dd', 'span', 'div'])
                if next_elem:
                    info['status'] = next_elem.get_text(strip=True)

            elif 'дата' in text and 'подач' in text:
                next_elem = label.find_next(['dd', 'span', 'div'])
                if next_elem:
                    try:
                        info['filing_date'] = datetime.strptime(
                            next_elem.get_text(strip=True),
                            '%d.%m.%Y'
                        )
                    except ValueError:
                        pass

        return info

    def _extract_parties(self, soup: BeautifulSoup) -> List[Party]:
        """
        Извлечение информации об участниках дела.
        """
        parties = []

        try:
            # Ищем блоки с участниками
            party_blocks = soup.find_all('div', class_=re.compile(r'party|participant|side'))

            for block in party_blocks:
                party = self._parse_party_block(block)
                if party:
                    parties.append(party)

        except Exception as e:
            logger.warning(f"Error extracting parties: {str(e)}")

        return parties

    def _parse_party_block(self, block) -> Optional[Party]:
        """
        Парсинг блока с информацией об участнике.
        """
        try:
            # Извлекаем имя
            name_elem = block.find(['span', 'div'], class_=re.compile(r'name|title'))
            if not name_elem:
                return None

            name = name_elem.get_text(strip=True)

            # Определяем роль
            role_text = block.get('class', [])
            role = 'plaintiff'  # по умолчанию

            if any('defendant' in cls or 'ответчик' in cls.lower() for cls in role_text):
                role = 'defendant'
            elif any('third' in cls or 'третье' in cls.lower() for cls in role_text):
                role = 'third_party'

            # Извлекаем дополнительные данные
            inn = None
            ogrn = None
            address = None

            # Ищем ИНН
            inn_elem = block.find(text=re.compile(r'ИНН|INN'))
            if inn_elem:
                inn_match = re.search(r'ИНН[:\s]*(\d+)', inn_elem, re.IGNORECASE)
                if inn_match:
                    inn = inn_match.group(1)

            # Ищем ОГРН
            ogrn_elem = block.find(text=re.compile(r'ОГРН|OGRN'))
            if ogrn_elem:
                ogrn_match = re.search(r'ОГРН[:\s]*(\d+)', ogrn_elem, re.IGNORECASE)
                if ogrn_match:
                    ogrn = ogrn_match.group(1)

            # Ищем адрес
            addr_elem = block.find(['span', 'div'], class_=re.compile(r'address|адрес'))
            if addr_elem:
                address = addr_elem.get_text(strip=True)

            return Party(
                name=name,
                inn=inn,
                ogrn=ogrn,
                address=address,
                role=role
            )

        except Exception as e:
            logger.warning(f"Error parsing party block: {str(e)}")
            return None

    def _extract_decisions(self, soup: BeautifulSoup) -> List[CourtDecision]:
        """
        Извлечение судебных решений.
        """
        decisions = []

        try:
            # Ищем блоки с судебными актами
            decision_blocks = soup.find_all('div', class_=re.compile(r'decision|judgment|акт'))

            for block in decision_blocks:
                decision = self._parse_decision_block(block)
                if decision:
                    decisions.append(decision)

        except Exception as e:
            logger.warning(f"Error extracting decisions: {str(e)}")

        return decisions

    def _parse_decision_block(self, block) -> Optional[CourtDecision]:
        """
        Парсинг блока с судебным решением.
        """
        try:
            # Извлекаем дату
            date_elem = block.find(['span', 'div'], class_=re.compile(r'date|дата'))
            decision_date = None
            if date_elem:
                try:
                    date_text = date_elem.get_text(strip=True)
                    decision_date = datetime.strptime(date_text, '%d.%m.%Y')
                except ValueError:
                    pass

            # Извлекаем тип решения
            type_elem = block.find(['span', 'div'], class_=re.compile(r'type|тип'))
            decision_type = type_elem.get_text(strip=True) if type_elem else 'decision'

            # Извлекаем резолютивную часть
            outcome_elem = block.find(['div', 'p'], class_=re.compile(r'outcome|результат'))
            outcome = outcome_elem.get_text(strip=True) if outcome_elem else None

            # Извлекаем сумму
            amount = None
            amount_match = re.search(r'(\d+(?:[.,]\d+)*)\s*(?:руб|рублей|руб\.)', str(block), re.IGNORECASE)
            if amount_match:
                amount_str = amount_match.group(1).replace(',', '.')
                try:
                    amount = float(amount_str)
                except ValueError:
                    pass

            return CourtDecision(
                date=decision_date or datetime.now(),
                court="",  # будет заполнено из основных данных дела
                decision_type=decision_type,
                outcome=outcome,
                amount=amount
            )

        except Exception as e:
            logger.warning(f"Error parsing decision block: {str(e)}")
            return None

    def _extract_claim_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Извлечение информации о сути иска.
        """
        claim_info = {}

        try:
            # Ищем блок с исковыми требованиями
            claim_block = soup.find('div', class_=re.compile(r'claim|иск|требования'))

            if claim_block:
                # Предмет иска
                subject_elem = claim_block.find(['h3', 'strong'], text=re.compile(r'предмет|требования'))
                if subject_elem:
                    next_elem = subject_elem.find_next(['p', 'div'])
                    if next_elem:
                        claim_info['subject'] = next_elem.get_text(strip=True)

                # Сумма иска
                amount_match = re.search(
                    r'сумм[аы]\s*(\d+(?:[.,]\d+)*)\s*(?:руб|рублей|руб\.)',
                    claim_block.get_text(),
                    re.IGNORECASE
                )
                if amount_match:
                    amount_str = amount_match.group(1).replace(',', '.')
                    try:
                        claim_info['amount'] = float(amount_str)
                    except ValueError:
                        pass

        except Exception as e:
            logger.warning(f"Error extracting claim info: {str(e)}")

        return claim_info

    def _classify_case_category(self, description: str, parties: List[Party], claim_info: Dict[str, Any]) -> CaseCategory:
        """
        Классификация категории судебного дела.
        """
        text = f"{description} {claim_info.get('subject', '')}".lower()

        if any(keyword in text for keyword in ['договор', 'контракт', 'соглашение', 'поставка', 'услуги']):
            return CaseCategory.CONTRACT

        elif any(keyword in text for keyword in ['имущество', 'собственность', 'недвижимость', 'земля']):
            return CaseCategory.PROPERTY

        elif any(keyword in text for keyword in ['акции', 'доля', 'участие', 'корпоративный']):
            return CaseCategory.CORPORATE

        elif any(keyword in text for keyword in ['налог', 'налоговый', 'фискальный']):
            return CaseCategory.TAX

        elif any(keyword in text for keyword in ['труд', 'работник', 'работодатель', 'увольнение']):
            return CaseCategory.LABOR

        elif any(keyword in text for keyword in ['административный', 'штраф', 'правонарушение']):
            return CaseCategory.ADMINISTRATIVE

        else:
            return CaseCategory.OTHER

    def _parse_case_status(self, status_text: str) -> CaseStatus:
        """
        Парсинг статуса судебного дела.
        """
        status_text = status_text.lower()

        if any(word in status_text for word in ['завершено', 'решено', 'окончено']):
            return CaseStatus.DECIDED

        elif any(word in status_text for word in ['производстве', 'рассмотрение']):
            return CaseStatus.PENDING

        elif any(word in status_text for word in ['обжаловано', 'апелляция']):
            return CaseStatus.APPEALED

        elif any(word in status_text for word in ['приостановлено', 'остановлено']):
            return CaseStatus.SUSPENDED

        elif any(word in status_text for word in ['закрыто', 'прекращено']):
            return CaseStatus.CLOSED

        else:
            return CaseStatus.PENDING  # по умолчанию
