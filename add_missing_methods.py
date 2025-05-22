
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    content = f.read()

# Добавляем методы перед методом get_recent_reports
new_methods = """
    def _generate_executive_summary(self, analysis_data: Dict[str, Any], 
                                   project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует исполнительное резюме отчета.
        
        Args:
            analysis_data: Данные анализа
            project_data: Данные проекта
            
        Returns:
            Dict[str, Any]: Данные исполнительного резюме
        """
        try:
            # Извлекаем ключевые метрики
            key_metrics = {
                'seo_score': analysis_data.get('seo_score', 0),
                'issues_found': len(analysis_data.get('issues', [])),
                'opportunities': len(analysis_data.get('opportunities', [])),
                'critical_issues': len([i for i in analysis_data.get('issues', []) if i.get('severity') == 'critical'])
            }
            
            # Формируем краткие выводы
            summary = {
                'overview': f"Анализ проекта '{project_data.get('name', 'Неизвестный')}' выявил {key_metrics['issues_found']} проблем и {key_metrics['opportunities']} возможностей для улучшения.",
                'seo_score': key_metrics['seo_score'],
                'key_findings': [
                    f"Общий SEO-рейтинг: {key_metrics['seo_score']}/100",
                    f"Критических проблем: {key_metrics['critical_issues']}",
                    f"Возможностей оптимизации: {key_metrics['opportunities']}"
                ],
                'recommendation_summary': "Рекомендуется в первую очередь устранить критические проблемы и воспользоваться выявленными возможностями."
            }
            
            print("✅ Исполнительное резюме сгенерировано")
            return summary
            
        except Exception as e:
            print(f"❌ Ошибка генерации исполнительного резюме: {e}")
            return {'error': str(e)}

    def _generate_recommendations_section(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует секцию рекомендаций.
        
        Args:
            analysis_data: Данные анализа
            
        Returns:
            Dict[str, Any]: Структурированные рекомендации
        """
        try:
            issues = analysis_data.get('issues', [])
            opportunities = analysis_data.get('opportunities', [])
            
            # Группируем рекомендации по приоритету
            recommendations = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }
            
            # Обрабатываем проблемы
            for issue in issues:
                priority = issue.get('severity', 'medium')
                if priority == 'critical':
                    recommendations['critical'].append({
                        'type': 'issue',
                        'title': issue.get('title', 'Неизвестная проблема'),
                        'description': issue.get('description', ''),
                        'action': issue.get('recommendation', 'Требует внимания')
                    })
                else:
                    recommendations.get(priority, recommendations['medium']).append({
                        'type': 'issue',
                        'title': issue.get('title', ''),
                        'description': issue.get('description', ''),
                        'action': issue.get('recommendation', '')
                    })
            
            # Обрабатываем возможности
            for opportunity in opportunities:
                priority = opportunity.get('impact', 'medium')
                recommendations.get(priority, recommendations['medium']).append({
                    'type': 'opportunity',
                    'title': opportunity.get('title', 'Возможность оптимизации'),
                    'description': opportunity.get('description', ''),
                    'action': opportunity.get('recommendation', 'Рекомендуется внедрить')
                })
            
            print("✅ Секция рекомендаций сгенерирована")
            return recommendations
            
        except Exception as e:
            print(f"❌ Ошибка генерации рекомендаций: {e}")
            return {'error': str(e)}

    def _apply_custom_styling(self, report_html: str, template: 'ReportTemplate') -> str:
        """
        Применяет пользовательские стили к отчету.
        
        Args:
            report_html: HTML-содержимое отчета
            template: Шаблон отчета с настройками стиля
            
        Returns:
            str: HTML с примененными стилями
        """
        try:
            # Получаем настройки стиля из шаблона
            style_settings = template.structure.get('styling', {})
            
            # Базовые стили
            css_styles = """
            <style>
                body { 
                    font-family: {font_family}; 
                    color: {text_color}; 
                    background-color: {bg_color}; 
                    margin: 20px;
                }
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
                .metric {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 10px; 
                    background-color: {card_bg}; 
                    border-radius: 5px; 
                }}
                .critical {{ color: #dc3545; font-weight: bold; }}
                .warning {{ color: #ffc107; }}
                .success {{ color: #28a745; }}
            </style>
            """.format(
                font_family=style_settings.get('font_family', 'Arial, sans-serif'),
                text_color=style_settings.get('text_color', '#333333'),
                bg_color=style_settings.get('background_color', '#ffffff'),
                accent_color=style_settings.get('accent_color', '#007bff'),
                card_bg=style_settings.get('card_background', '#f8f9fa')
            )
            
            # Добавляем стили в начало HTML
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
        """
        Добавляет интерактивные элементы в отчет.
        
        Args:
            report_html: HTML-содержимое отчета
            report_data: Данные отчета для интерактивных элементов
            
        Returns:
            str: HTML с интерактивными элементами
        """
        try:
            # JavaScript для интерактивности
            interactive_js = """
            <script>
                // Переключение видимости секций
                function toggleSection(sectionId) {{
                    const element = document.getElementById(sectionId);
                    if (element.style.display === 'none') {{
                        element.style.display = 'block';
                    }} else {{
                        element.style.display = 'none';
                    }}
                }}
                
                // Фильтрация рекомендаций по приоритету
                function filterRecommendations(priority) {{
                    const recommendations = document.querySelectorAll('.recommendation');
                    recommendations.forEach(rec => {{
                        if (priority === 'all' || rec.classList.contains(priority)) {{
                            rec.style.display = 'block';
                        }} else {{
                            rec.style.display = 'none';
                        }}
                    }});
                }}
                
                // Экспорт данных
                function exportData(format) {{
                    console.log('Экспорт в формате:', format);
                    alert('Функция экспорта будет реализована в следующих версиях');
                }}
                
                // Инициализация при загрузке страницы
                document.addEventListener('DOMContentLoaded', function() {{
                    console.log('Интерактивный отчет загружен');
                }});
            </script>
            """
            
            # Добавляем JavaScript в конец HTML
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

print("✅ Все недостающие методы из чек-листа добавлены!")
