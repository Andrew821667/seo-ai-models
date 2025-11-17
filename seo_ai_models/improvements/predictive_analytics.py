"""
Predictive Analytics - Прогнозирование SEO метрик и трендов.

Функции:
- Прогноз трафика
- Предсказание позиций
- Trend analysis
- ROI forecasting
- Seasonal pattern detection
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class PredictiveAnalytics:
    """Предиктивная аналитика для SEO."""

    def __init__(self, analytics_connector=None):
        self.analytics = analytics_connector
        self.historical_data = {}
        logger.info("PredictiveAnalytics initialized")

    def forecast_traffic(self, domain: str, months_ahead: int = 3) -> Dict[str, Any]:
        """Прогнозирует органический трафик."""
        # Получаем исторические данные
        historical = self._get_historical_traffic(domain, months=12)

        if not historical:
            return {
                "error": "Insufficient historical data",
                "recommendation": "Need at least 6 months of data",
            }

        # Простое линейное прогнозирование
        forecast = {
            "domain": domain,
            "forecast_period": f"{months_ahead} months",
            "predictions": [],
            "confidence": 0,
            "trend": "",
            "growth_rate": 0,
        }

        # Рассчитываем тренд
        values = [d["traffic"] for d in historical]
        trend_slope = self._calculate_trend(values)

        # Прогнозируем
        last_value = values[-1]

        for i in range(1, months_ahead + 1):
            predicted = last_value + (trend_slope * i)
            predicted = max(0, predicted)  # Не может быть отрицательным

            forecast["predictions"].append(
                {
                    "month": i,
                    "predicted_traffic": round(predicted),
                    "lower_bound": round(predicted * 0.8),
                    "upper_bound": round(predicted * 1.2),
                }
            )

        # Определяем тренд
        if trend_slope > 100:
            forecast["trend"] = "strong_growth"
        elif trend_slope > 0:
            forecast["trend"] = "moderate_growth"
        elif trend_slope > -100:
            forecast["trend"] = "moderate_decline"
        else:
            forecast["trend"] = "strong_decline"

        # Рассчитываем growth rate
        if values[0] > 0:
            total_growth = ((values[-1] - values[0]) / values[0]) * 100
            forecast["growth_rate"] = round(total_growth, 1)

        # Confidence score
        forecast["confidence"] = self._calculate_forecast_confidence(historical)

        logger.info(f"Traffic forecast: {forecast['trend']}, {forecast['growth_rate']}% growth")

        return forecast

    def predict_ranking_changes(self, keyword: str, current_position: int) -> Dict[str, Any]:
        """Прогнозирует изменения позиций."""
        prediction = {
            "keyword": keyword,
            "current_position": current_position,
            "predicted_position": 0,
            "probability": 0,
            "timeframe": "3 months",
            "factors": [],
        }

        # Анализируем факторы
        factors = []

        # Фактор 1: Content quality
        content_score = self._assess_content_optimization(keyword)
        if content_score > 0.7:
            factors.append(
                {"factor": "Content Quality", "impact": "+3 positions", "confidence": "high"}
            )

        # Фактор 2: Backlinks
        backlinks_trend = self._analyze_backlinks_trend()
        if backlinks_trend == "growing":
            factors.append(
                {"factor": "Growing Backlinks", "impact": "+2 positions", "confidence": "medium"}
            )

        # Фактор 3: Competition
        competition_level = self._assess_competition(keyword)
        if competition_level == "high":
            factors.append(
                {"factor": "High Competition", "impact": "-1 position", "confidence": "medium"}
            )

        # Рассчитываем предсказанную позицию
        position_change = sum(
            [
                3 if "Content Quality" in str(factors) else 0,
                2 if "Growing Backlinks" in str(factors) else 0,
                -1 if "High Competition" in str(factors) else 0,
            ]
        )

        predicted_position = max(1, current_position - position_change)

        prediction["predicted_position"] = predicted_position
        prediction["factors"] = factors
        prediction["probability"] = 0.65  # 65% confidence

        logger.info(
            f"Ranking prediction for '{keyword}': {current_position} → {predicted_position}"
        )

        return prediction

    def detect_trends(self, domain: str, metric: str = "traffic") -> Dict[str, Any]:
        """Обнаруживает тренды в данных."""
        data = self._get_historical_data(domain, metric)

        trends = {
            "metric": metric,
            "timeframe": "last 12 months",
            "detected_trends": [],
            "seasonality": None,
            "anomalies": [],
        }

        if not data:
            return trends

        values = [d["value"] for d in data]

        # Детектируем сезонность
        seasonality = self._detect_seasonality(values)
        if seasonality:
            trends["seasonality"] = seasonality

        # Детектируем аномалии
        anomalies = self._detect_anomalies(values)
        trends["anomalies"] = anomalies

        # Общие тренды
        if self._is_growing(values):
            trends["detected_trends"].append(
                {
                    "type": "growth",
                    "strength": "strong" if self._calculate_trend(values) > 100 else "moderate",
                    "description": f"{metric.title()} is consistently growing",
                }
            )

        if self._is_volatile(values):
            trends["detected_trends"].append(
                {
                    "type": "volatility",
                    "strength": "high",
                    "description": f"{metric.title()} shows high volatility",
                }
            )

        logger.info(f"Detected {len(trends['detected_trends'])} trends for {metric}")

        return trends

    def calculate_roi_forecast(self, investment: float, strategy: str) -> Dict[str, Any]:
        """Прогнозирует ROI от SEO стратегии."""
        roi_forecast = {
            "investment": investment,
            "strategy": strategy,
            "timeframe": "12 months",
            "projected_revenue": 0,
            "projected_roi": 0,
            "monthly_breakdown": [],
            "assumptions": [],
        }

        # Базовые коэффициенты в зависимости от стратегии
        strategy_multipliers = {
            "content_marketing": 3.5,
            "link_building": 4.0,
            "technical_seo": 2.5,
            "local_seo": 5.0,
            "comprehensive": 6.0,
        }

        multiplier = strategy_multipliers.get(strategy, 3.0)

        # Рассчитываем проекцию по месяцам
        total_revenue = 0

        for month in range(1, 13):
            # ROI растет со временем (экспоненциально)
            monthly_multiplier = multiplier * (1 + (month / 12) * 0.5)
            monthly_revenue = (investment / 12) * monthly_multiplier

            total_revenue += monthly_revenue

            roi_forecast["monthly_breakdown"].append(
                {
                    "month": month,
                    "revenue": round(monthly_revenue, 2),
                    "cumulative_revenue": round(total_revenue, 2),
                    "cumulative_roi": round(((total_revenue - investment) / investment) * 100, 1),
                }
            )

        roi_forecast["projected_revenue"] = round(total_revenue, 2)
        roi_forecast["projected_roi"] = round(((total_revenue - investment) / investment) * 100, 1)

        # Assumptions
        roi_forecast["assumptions"] = [
            "Assumes consistent effort throughout the period",
            "Based on industry averages",
            "Actual results may vary based on competition and niche",
            f"Average conversion rate: 2-3%",
            "Organic traffic growth: 15-30% quarterly",
        ]

        logger.info(f"ROI forecast: {roi_forecast['projected_roi']}% for {strategy}")

        return roi_forecast

    def identify_seasonal_patterns(self, domain: str, years: int = 2) -> Dict[str, Any]:
        """Определяет сезонные паттерны в трафике."""
        historical = self._get_historical_traffic(domain, months=years * 12)

        patterns = {
            "domain": domain,
            "analysis_period": f"{years} years",
            "peak_months": [],
            "low_months": [],
            "seasonal_index": {},
            "recommendations": [],
        }

        if not historical or len(historical) < 12:
            return {
                "error": "Insufficient data for seasonal analysis",
                "required": "At least 12 months of data",
            }

        # Группируем по месяцам
        monthly_averages = {}

        for entry in historical:
            month = entry["date"].month
            if month not in monthly_averages:
                monthly_averages[month] = []
            monthly_averages[month].append(entry["traffic"])

        # Рассчитываем средние
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        for month in range(1, 13):
            if month in monthly_averages:
                avg = statistics.mean(monthly_averages[month])
                patterns["seasonal_index"][month_names[month - 1]] = round(avg)

        # Находим пики и спады
        if patterns["seasonal_index"]:
            values = list(patterns["seasonal_index"].values())
            max_traffic = max(values)
            min_traffic = min(values)

            for month_name, traffic in patterns["seasonal_index"].items():
                if traffic == max_traffic:
                    patterns["peak_months"].append(month_name)
                if traffic == min_traffic:
                    patterns["low_months"].append(month_name)

        # Recommendations
        if patterns["peak_months"]:
            patterns["recommendations"].append(
                f"Increase content production before {', '.join(patterns['peak_months'])} to capture peak traffic"
            )

        if patterns["low_months"]:
            patterns["recommendations"].append(
                f"Use {', '.join(patterns['low_months'])} for technical improvements and content planning"
            )

        logger.info(f"Seasonal analysis: peaks in {patterns['peak_months']}")

        return patterns

    # Helper methods

    def _get_historical_traffic(self, domain: str, months: int = 12) -> List[Dict]:
        """Получает исторические данные трафика."""
        # В реальности получать из Analytics API
        # Генерируем фейковые данные для примера
        data = []
        base_traffic = 10000

        for i in range(months):
            date = datetime.now() - timedelta(days=30 * (months - i))
            traffic = base_traffic + (i * 500) + ((-1) ** i * 200)  # Рост с флуктуациями

            data.append({"date": date, "traffic": max(0, int(traffic))})

        return data

    def _get_historical_data(self, domain: str, metric: str) -> List[Dict]:
        """Получает исторические данные по метрике."""
        # Заглушка
        return []

    def _calculate_trend(self, values: List[float]) -> float:
        """Рассчитывает тренд (slope)."""
        if len(values) < 2:
            return 0

        n = len(values)
        x = list(range(n))

        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0

        slope = numerator / denominator
        return slope

    def _calculate_forecast_confidence(self, historical: List[Dict]) -> float:
        """Рассчитывает уверенность в прогнозе."""
        if len(historical) < 6:
            return 0.4  # Низкая уверенность

        values = [d["traffic"] for d in historical]

        # Проверяем стабильность
        if len(values) > 1:
            std_dev = statistics.stdev(values)
            mean = statistics.mean(values)

            coefficient_of_variation = std_dev / mean if mean > 0 else 1

            # Чем меньше вариация, тем выше уверенность
            confidence = max(0.5, min(0.95, 1 - coefficient_of_variation))
            return round(confidence, 2)

        return 0.6

    def _assess_content_optimization(self, keyword: str) -> float:
        """Оценивает оптимизацию контента."""
        # Упрощенная оценка
        return 0.75

    def _analyze_backlinks_trend(self) -> str:
        """Анализирует тренд backlinks."""
        # Заглушка
        return "growing"

    def _assess_competition(self, keyword: str) -> str:
        """Оценивает уровень конкуренции."""
        # Заглушка
        return "medium"

    def _detect_seasonality(self, values: List[float]) -> Dict:
        """Детектирует сезонность."""
        if len(values) < 12:
            return None

        # Упрощенная детекция
        return {"detected": True, "period": "12 months", "strength": "moderate"}

    def _detect_anomalies(self, values: List[float]) -> List[Dict]:
        """Детектирует аномалии."""
        if len(values) < 3:
            return []

        anomalies = []
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0

        for i, value in enumerate(values):
            z_score = abs((value - mean) / std_dev) if std_dev > 0 else 0

            if z_score > 2:  # Более 2 стандартных отклонений
                anomalies.append(
                    {
                        "index": i,
                        "value": value,
                        "type": "spike" if value > mean else "drop",
                        "severity": "high" if z_score > 3 else "medium",
                    }
                )

        return anomalies

    def _is_growing(self, values: List[float]) -> bool:
        """Проверяет, растут ли значения."""
        if len(values) < 3:
            return False

        slope = self._calculate_trend(values)
        return slope > 0

    def _is_volatile(self, values: List[float]) -> bool:
        """Проверяет, волатильны ли значения."""
        if len(values) < 2:
            return False

        std_dev = statistics.stdev(values)
        mean = statistics.mean(values)

        coefficient_of_variation = std_dev / mean if mean > 0 else 0

        return coefficient_of_variation > 0.3  # > 30% variation
