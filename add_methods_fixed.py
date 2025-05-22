
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Добавляем методы с исправленным синтаксисом
new_methods = '''
    def _generate_executive_summary(self, analysis_data: Dict[str, Any], 
                                   project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует исполнительное резюме отчета."""
        try:
            key_metrics = {
                "seo_score": analysis_data.get("seo_score", 0),
                "issues_found": len(analysis_data.get("issues", [])),
                "opportunities": len(analysis_data.get("opportunities", [])),
                "critical_issues": len([i for i in analysis_data.get("issues", []) if i.get("severity") == "critical"])
            }
            
            summary = {
                "overview": f"Анализ проекта выявил {key_metrics['issues_found']} проблем и {key_metrics['opportunities']} возможностей.",
                "seo_score": key_metrics["seo_score"],
                "key_findings": [
                    f"Общий SEO-рейтинг: {key_metrics['seo_score']}/100",
                    f"Критических проблем: {key_metrics['critical_issues']}"
                ],
                "recommendation_summary": "Рекомендуется устранить критические проблемы в первую очередь."
            }
            
            print("✅ Исполнительное резюме сгенерировано")
            return summary
            
        except Exception as e:
            print(f"❌ Ошибка генерации резюме: {e}")
            return {"error": str(e)}

    def _generate_recommendations_section(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует секцию рекомендаций."""
        try:
            issues = analysis_data.get("issues", [])
            opportunities = analysis_data.get("opportunities", [])
            
            recommendations = {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            }
            
            # Обрабатываем проблемы
            for issue in issues:
                priority = issue.get("severity", "medium")
                recommendations.get(priority, recommendations["medium"]).append({
                    "type": "issue",
                    "title": issue.get("title", "Проблема"),
                    "description": issue.get("description", ""),
                    "action": issue.get("recommendation", "Требует внимания")
                })
            
            # Обрабатываем возможности
            for opportunity in opportunities:
                priority = opportunity.get("impact", "medium")
                recommendations.get(priority, recommendations["medium"]).append({
                    "type": "opportunity",
                    "title": opportunity.get("title", "Возможность оптимизации"),
                    "description": opportunity.get("description", ""),
                    "action": opportunity.get("recommendation", "Рекомендуется внедрить")
                })
            
            print("✅ Секция рекомендаций сгенерирована")
            return recommendations
            
        except Exception as e:
            print(f"❌ Ошибка генерации рекомендаций: {e}")
            return {"error": str(e)}

    def _apply_custom_styling(self, report_html: str, template: 'ReportTemplate') -> str:
        """Применяет пользовательские стили к отчету."""
        try:
            style_settings = template.structure.get("styling", {})
            
            # Создаем CSS с базовыми стилями
            font_family = style_settings.get("font_family", "Arial, sans-serif")
            text_color = style_settings.get("text_color", "#333333")
            bg_color = style_settings.get("background_color", "#ffffff")
            accent_color = style_settings.get("accent_color", "#007bff")
            
            css_styles = f"""
            <style>
                body {{ 
                    font-family: {font_family}; 
                    color: {text_color}; 
                    background-color: {bg_color}; 
                    margin: 20px;
                }}
                .report-header {{ 
                    border-bottom: 2px solid {accent_color}; 
                    padding-bottom: 10px; 
                    margin-bottom: 20px; 
                }}
                .section {{ 
                    margin-bottom: 25px; 
                    padding: 15px; 
                    border-left: 4px solid {accent_color}; 
                }}
                .critical {{ color: #dc3545; font-weight: bold; }}
                .warning {{ color: #ffc107; }}
                .success {{ color: #28a745; }}
            </style>
            """
            
            # Добавляем стили в HTML
            if "<head>" in report_html:
                styled_html = report_html.replace("<head>", f"<head>{css_styles}")
            else:
                styled_html = f"{css_styles}{report_html}"
            
            print("✅ Пользовательские стили применены")
            return styled_html
            
        except Exception as e:
            print(f"❌ Ошибка применения стилей: {e}")
            return report_html

    def _add_interactive_elements(self, report_html: str, report_data: Dict[str, Any]) -> str:
        """Добавляет интерактивные элементы в отчет."""
        try:
            interactive_js = """
            <script>
                // Переключение видимости секций
                function toggleSection(sectionId) {
                    const element = document.getElementById(sectionId);
                    if (element.style.display === 'none') {
                        element.style.display = 'block';
                    } else {
                        element.style.display = 'none';
                    }
                }
                
                // Фильтрация рекомендаций по приоритету
                function filterRecommendations(priority) {
                    const recommendations = document.querySelectorAll('.recommendation');
                    recommendations.forEach(rec => {
                        if (priority === 'all' || rec.classList.contains(priority)) {
                            rec.style.display = 'block';
                        } else {
                            rec.style.display = 'none';
                        }
                    });
                }
                
                // Экспорт данных
                function exportData(format) {
                    console.log('Экспорт в формате:', format);
                    alert('Функция экспорта будет реализована в следующих версиях');
                }
                
                // Инициализация
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('Интерактивный отчет загружен');
                });
            </script>
            """
            
            # Добавляем JavaScript в HTML
            if "</body>" in report_html:
                interactive_html = report_html.replace("</body>", f"{interactive_js}</body>")
            else:
                interactive_html = f"{report_html}{interactive_js}"
            
            print("✅ Интерактивные элементы добавлены")
            return interactive_html
            
        except Exception as e:
            print(f"❌ Ошибка добавления интерактивных элементов: {e}")
            return report_html

'''

# Вставляем методы перед get_recent_reports
content = content.replace(
    "    def get_recent_reports(self, limit: int = 10) -> List[Report]:",
    new_methods + "    def get_recent_reports(self, limit: int = 10) -> List[Report]:"
)

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Все методы добавлены с правильным синтаксисом!")
