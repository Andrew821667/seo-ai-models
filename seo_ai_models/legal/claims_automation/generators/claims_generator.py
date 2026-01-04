"""
Генератор претензий и исковых заявлений.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from seo_ai_models.legal.claims_automation.models.claims_models import (
    Claim, CourtClaim, Party, ClaimType, CourtClaimType, LimitationPeriodType
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClaimsGenerator:
    """
    Генератор текстов претензий и исковых заявлений.
    """

    def __init__(self, templates_path: Optional[str] = None):
        """Инициализация генератора."""
        self.templates_path = Path(templates_path) if templates_path else Path(__file__).parent / "templates"
        self.templates_path.mkdir(parents=True, exist_ok=True)

        # Базовые шаблоны
        self.templates = self._load_templates()

    def generate_claim(self, claim: Claim) -> str:
        """
        Генерация текста претензии.

        Args:
            claim: Данные претензии

        Returns:
            str: Сформированный текст претензии
        """
        try:
            template = self.templates.get('claim', self._get_default_claim_template())

            # Заполняем шаблон данными
            claim_text = template.format(
                claimant_name=claim.claimant.name,
                claimant_address=claim.claimant.address,
                claimant_inn=f"ИНН: {claim.claimant.inn}" if claim.claimant.inn else "",
                claimant_ogrn=f"ОГРН: {claim.claimant.ogrn}" if claim.claimant.ogrn else "",
                claimant_representative=claim.claimant.representative or "________________",

                respondent_name=claim.respondent.name,
                respondent_address=claim.respondent.address,
                respondent_inn=f"ИНН: {claim.respondent.inn}" if claim.respondent.inn else "",
                respondent_ogrn=f"ОГРН: {claim.respondent.ogrn}" if claim.respondent.ogrn else "",

                claim_title=claim.title,
                claim_description=claim.description,
                claim_amount=self._format_amount(claim.amount, claim.currency) if claim.amount else "",
                due_date=claim.due_date.strftime("%d.%m.%Y") if claim.due_date else "10 рабочих дней",

                current_date=datetime.now().strftime("%d.%m.%Y"),
                claim_id=claim.claim_id
            )

            return claim_text

        except Exception as e:
            logger.error(f"Error generating claim text: {str(e)}")
            return self._generate_fallback_claim_text(claim)

    def generate_court_claim(self, court_claim: CourtClaim) -> str:
        """
        Генерация текста искового заявления.

        Args:
            court_claim: Данные искового заявления

        Returns:
            str: Сформированный текст искового заявления
        """
        try:
            template = self.templates.get('court_claim', self._get_default_court_claim_template())

            # Формируем список третьих лиц
            third_parties_text = ""
            if court_claim.third_parties:
                third_parties_list = []
                for i, party in enumerate(court_claim.third_parties, 1):
                    third_parties_list.append(f"{i}. {party.name}, {party.address}")
                third_parties_text = "\n".join(third_parties_list)

            # Формируем список доказательств
            evidence_text = ""
            if court_claim.evidence:
                evidence_list = []
                for i, evidence in enumerate(court_claim.evidence, 1):
                    evidence_list.append(f"{i}. {evidence.get('description', '')}")
                evidence_text = "\n".join(evidence_list)

            # Формируем ссылки на нормы права
            legal_refs_text = ""
            if court_claim.legal_references:
                legal_refs_text = "\n".join(f"- {ref}" for ref in court_claim.legal_references)

            # Заполняем шаблон
            claim_text = template.format(
                court_name="Арбитражный суд города Москвы",  # можно параметризировать
                claimant_name=court_claim.claimant.name,
                claimant_address=court_claim.claimant.address,
                claimant_inn=court_claim.claimant.inn or "",
                claimant_ogrn=court_claim.claimant.ogrn or "",
                claimant_representative=court_claim.claimant.representative or "________________",

                respondent_name=court_claim.respondent.name,
                respondent_address=court_claim.respondent.address,
                respondent_inn=court_claim.respondent.inn or "",
                respondent_ogrn=court_claim.respondent.ogrn or "",
                respondent_representative=court_claim.respondent.representative or "________________",

                third_parties=third_parties_text,

                claim_title=court_claim.title,
                claim_subject=court_claim.claim_subject,
                claim_amount=self._format_amount(court_claim.claim_amount, court_claim.currency) if court_claim.claim_amount else "",
                circumstances=court_claim.circumstances,
                evidence=evidence_text,
                legal_references=legal_refs_text,
                calculation=court_claim.calculation or "",

                current_date=datetime.now().strftime("%d.%m.%Y"),
                claim_id=court_claim.claim_id
            )

            return claim_text

        except Exception as e:
            logger.error(f"Error generating court claim text: {str(e)}")
            return self._generate_fallback_court_claim_text(court_claim)

    def generate_response_to_claim(self, original_claim: Claim, response_type: str, response_text: str) -> str:
        """
        Генерация ответа на претензию.

        Args:
            original_claim: Оригинальная претензия
            response_type: Тип ответа (satisfaction, partial_satisfaction, rejection)
            response_text: Текст ответа

        Returns:
            str: Сформированный текст ответа
        """
        try:
            template = self.templates.get('claim_response', self._get_default_response_template())

            response_type_text = {
                'satisfaction': 'Удовлетворение претензии',
                'partial_satisfaction': 'Частичное удовлетворение претензии',
                'rejection': 'Отклонение претензии'
            }.get(response_type, 'Ответ на претензию')

            response_text_formatted = template.format(
                respondent_name=original_claim.respondent.name,
                respondent_address=original_claim.respondent.address,
                respondent_representative=original_claim.respondent.representative or "________________",

                claimant_name=original_claim.claimant.name,
                claimant_address=original_claim.claimant.address,

                original_claim_date=original_claim.sent_at.strftime("%d.%m.%Y") if original_claim.sent_at else "",
                original_claim_title=original_claim.title,

                response_type=response_type_text,
                response_text=response_text,

                current_date=datetime.now().strftime("%d.%m.%Y")
            )

            return response_text_formatted

        except Exception as e:
            logger.error(f"Error generating claim response: {str(e)}")
            return f"Ответ на претензию {original_claim.claim_id}\n\n{response_text}"

    def _load_templates(self) -> Dict[str, str]:
        """
        Загрузка шаблонов документов.
        """
        templates = {}

        # Пытаемся загрузить шаблоны из файлов
        template_files = {
            'claim': 'claim_template.txt',
            'court_claim': 'court_claim_template.txt',
            'claim_response': 'claim_response_template.txt'
        }

        for template_name, filename in template_files.items():
            template_path = self.templates_path / filename
            try:
                if template_path.exists():
                    with open(template_path, 'r', encoding='utf-8') as f:
                        templates[template_name] = f.read()
                    logger.info(f"Loaded template: {filename}")
                else:
                    logger.warning(f"Template file not found: {filename}")
            except Exception as e:
                logger.error(f"Error loading template {filename}: {str(e)}")

        return templates

    def _get_default_claim_template(self) -> str:
        """
        Получение шаблона претензии по умолчанию.
        """
        return """ПРЕТЕНЗИЯ № {claim_id}

{current_date}

{claimant_name}
{claimant_address}
{claimant_inn}
{claimant_ogrn}

{claimant_representative}

{respondent_name}
{respondent_address}
{respondent_inn}
{respondent_ogrn}

Уважаемые господа!

Направляем Вам претензию по факту: {claim_title}

{claim_description}

{claim_amount}

Требуем в срок до {due_date} удовлетворить наши требования.

В случае неудовлетворения претензии в указанный срок будем вынуждены обратиться в суд.

С уважением,
{claimant_representative}
"""

    def _get_default_court_claim_template(self) -> str:
        """
        Получение шаблона искового заявления по умолчанию.
        """
        return """В {court_name}

Истец: {claimant_name}
{claimant_address}
ИНН: {claimant_inn}
ОГРН: {claimant_ogrn}

Представитель истца: {claimant_representative}

Ответчик: {respondent_name}
{respondent_address}
ИНН: {respondent_inn}
ОГРН: {respondent_ogrn}

Представитель ответчика: {respondent_representative}

Третьи лица:
{third_parties}

ИСКОВОЕ ЗАЯВЛЕНИЕ
{claim_title}

Цена иска: {claim_amount}

{current_date}

Обстоятельства дела:
{circumstances}

Доказательства:
{evidence}

Нормы права:
{legal_references}

Расчет исковых требований:
{calculation}

На основании изложенного, руководствуясь ст. 125, 126 АПК РФ,

ПРОШУ:

1. Принять исковое заявление к производству суда.
2. Взыскать с ответчика в пользу истца {claim_amount}.
3. Взыскать с ответчика судебные расходы.

Приложения:
1. Документы, подтверждающие основания иска.
2. Доверенность на представителя.
3. Документы, подтверждающие направление иска ответчику.

{claimant_representative}
"""

    def _get_default_response_template(self) -> str:
        """
        Получение шаблона ответа на претензию по умолчанию.
        """
        return """{respondent_name}
{respondent_address}

{respondent_representative}

{claimant_name}
{claimant_address}

{current_date}

ОТВЕТ НА ПРЕТЕНЗИЮ
от {original_claim_date}
"{original_claim_title}"

Уважаемые господа!

В ответ на Вашу претензию от {original_claim_date} сообщаем следующее:

{response_type}

{response_text}

С уважением,
{respondent_representative}
"""

    def _format_amount(self, amount: float, currency: str) -> str:
        """
        Форматирование суммы.
        """
        if currency == "RUB":
            return f"{amount:,.2f} руб."
        elif currency == "USD":
            return f"${amount:,.2f}"
        elif currency == "EUR":
            return f"€{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"

    def _generate_fallback_claim_text(self, claim: Claim) -> str:
        """
        Генерация текста претензии в случае ошибки.
        """
        return f"""ПРЕТЕНЗИЯ № {claim.claim_id}

{datetime.now().strftime('%d.%m.%Y')}

От: {claim.claimant.name}
Кому: {claim.respondent.name}

{claim.title}

{claim.description}

Сумма: {self._format_amount(claim.amount, claim.currency) if claim.amount else 'Не указана'}

С уважением,
{claim.claimant.representative or claim.claimant.name}
"""

    def _generate_fallback_court_claim_text(self, court_claim: CourtClaim) -> str:
        """
        Генерация текста искового заявления в случае ошибки.
        """
        return f"""ИСКОВОЕ ЗАЯВЛЕНИЕ № {court_claim.claim_id}

{datetime.now().strftime('%d.%m.%Y')}

Истец: {court_claim.claimant.name}
Ответчик: {court_claim.respondent.name}

{court_claim.title}

Предмет иска: {court_claim.claim_subject}

{court_claim.circumstances}

С уважением,
{court_claim.claimant.representative or court_claim.claimant.name}
"""
