
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Создаем методы с правильными отступами
new_methods = """
    def _generate_executive_summary(self, analysis_data: Dict[str, Any], 
                                   project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует исполнительное резюме отчета."""
        try:
            key_metrics = {
                'seo_score': analysis_data.get('seo_score', 0),
                'issues_found': len(analysis_data.get('issues', [])),
                'opportunities': len(analysis_data.get('opportunities', [])),
                'critical_issues': len([i for i in analysis_data.get('issues', []) if i.get('severity') == 'critical'])
            }
            
            summary = {
                'overview': f"Анализ проекта '{project_data.get('name', 'Неизвестный')}' выявил {key_metrics['issues_found']} проблем.",
                'seo_score': key_metrics['seo_score'],
                'key_findings': [
                    f"Общий SEO-рейтинг: {key_metrics['seo_score']}/100",
                    f"Критических проблем: {key_metrics['critical_issues']}"
                ]
            }
            
            print("✅ Исполнительное резюме сгенерировано")
            return summary
            
        except Exception as e:
            print(f"❌ Ошибка генерации резюме: {e}")
            return {'error': str(e)}

    def _generate_recommendations_section(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует секцию рекомендаций."""
        try:
            issues = analysis_data.get('issues', [])
            opportunities = analysis_data.get('opportunities', [])
            
            recommendations = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }
            
            for issue in issues:
                priority = issue.get('severity', 'medium')
                recommendations.get(priority, recommendations['medium']).append({
                    'type': 'issue',
                    'title': issue.get('title', 'Проблема'),
                    'action': issue.get('recommendation', 'Требует внимания')
                })
            
            print("✅ Секция рекомендаций сгенерирована")
            return recommendations
            
        except Exception as e:
            print(f"❌ Ошибка генерации рекомендаций: {e}")
            return {'error': str(e)}

    def _apply_custom_styling(self, report_html: str, template: 'ReportTemplate') -> str:
        """Применяет пользовательские стили к отчету."""
        try:
            style_settings = template.structure.get('styling', {})
            
            css_styles = f"""
            <style>
                body {{ font-family: {style_settings.get('font_family', 'Arial, sans-serif')}; }}
                .report-header {{ border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                .section {{ margin-bottom: 25px; padding: 15px; }}
                .critical {{ color: #dc3545; font-weight: bold; }}
            </style>
            """
            
            if '<head>' in report_html:
                styled_html = report_html.replace('<head>', f'<head>{css_styles}')
            else:
                styled_html = f'{css_styles}{report_html}'
            
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
                function toggleSection(sectionId) {
                    const element = document.getElementById(sectionId);
                    element.style.display = element.style.display === 'none' ? 'block' : 'none';
                }
                
                function filterRecommendations(priority) {
                    const recommendations = document.querySelectorAll('.recommendation');
                    recommendations.forEach(rec => {
                        rec.style.display = (priority === 'all' || rec.classList.contains(priority)) ? 'block' : 'none';
                    });
                }
            </script>
            """
            
            if '</body>' in report_html:
                interactive_html = report_html.replace('</body>', f'{interactive_js}</body>')
            else:
                interactive_html = f'{report_html}{interactive_js}'
            
            print("✅ Интерактивные элементы добавлены")
            return interactive_html
            
        except Exception as e:
            print(f"❌ Ошибка добавления интерактивных элементов: {e}")
            return report_html

"""

# Вставляем методы перед get_recent_reports
content = content.replace(
    "    def get_recent_reports(self, limit: int = 10) -> List[Report]:",
    new_methods + "    def get_recent_reports(self, limit: int = 10) -> List[Report]:"
)

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Все методы из чек-листа добавлены!")
