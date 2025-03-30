"""
Инструмент для сбора и автоматической разметки контента с веб-сайтов
для обучения модели E-E-A-T на реальных данных
"""
import requests
from bs4 import BeautifulSoup
import json
import csv
import random
import time
import os
import re
import sys  # Added the missing import
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm

# Импортируем наш анализатор E-E-A-T
sys.path.append('/content/seo-ai-models')
from models.seo_advisor.enhanced_eeat_analyzer import EnhancedEEATAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WebContentCollector:
    """Класс для сбора веб-контента и его автоматической разметки"""
    
    def __init__(self, output_dir: str = "collected_data", model_path: Optional[str] = None):
        """
        Инициализация коллектора
        
        Args:
            output_dir: Директория для сохранения собранных данных
            model_path: Путь к обученной модели E-E-A-T (опционально)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Инициализация анализатора E-E-A-T
        self.analyzer = EnhancedEEATAnalyzer(model_path=model_path)
        
        # Заголовки для имитации браузера
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Категории YMYL
        self.ymyl_categories = {
            'finance': ['money', 'invest', 'banking', 'loan', 'credit', 'tax', 'insurance'],
            'health': ['health', 'medical', 'disease', 'treatment', 'medicine', 'doctor', 'symptom'],
            'legal': ['law', 'legal', 'attorney', 'lawyer', 'rights', 'lawsuit', 'court']
        }
        
        # Категории не-YMYL
        self.non_ymyl_categories = {
            'tech': ['technology', 'tech', 'software', 'programming', 'developer', 'code'],
            'travel': ['travel', 'vacation', 'tourism', 'destination', 'trip', 'hotel'],
            'blog': ['blog', 'lifestyle', 'personal', 'story', 'experience'],
            'entertainment': ['entertainment', 'movie', 'music', 'game', 'sport', 'fun']
        }
    
    def identify_category(self, url: str, content: str) -> Tuple[str, bool]:
        """
        Определение категории и YMYL-статуса контента
        
        Args:
            url: URL страницы
            content: Текстовое содержимое страницы
            
        Returns:
            Кортеж (категория, is_ymyl)
        """
        domain = urlparse(url).netloc
        
        # Проверка на известные авторитетные домены по категориям
        known_domains = {
            # YMYL домены
            'finance': ['investopedia.com', 'bloomberg.com', 'wsj.com', 'finance.yahoo.com', 'cnbc.com'],
            'health': ['mayoclinic.org', 'webmd.com', 'nih.gov', 'who.int', 'healthline.com'],
            'legal': ['law.com', 'findlaw.com', 'justia.com', 'lawyers.com', 'nolo.com'],
            
            # Не-YMYL домены
            'tech': ['techcrunch.com', 'wired.com', 'theverge.com', 'cnet.com', 'zdnet.com'],
            'travel': ['tripadvisor.com', 'lonelyplanet.com', 'booking.com', 'expedia.com'],
            'blog': ['medium.com', 'wordpress.com', 'blogspot.com', 'substack.com'],
            'entertainment': ['variety.com', 'ign.com', 'imdb.com', 'rottentomatoes.com']
        }
        
        # Проверка домена
        for category, domains in known_domains.items():
            if any(d in domain for d in domains):
                is_ymyl = category in ['finance', 'health', 'legal']
                return category, is_ymyl
        
        # Анализ ключевых слов в содержимом
        content_lower = content.lower()
        category_scores = {}
        
        # Подсчет упоминаний ключевых слов для YMYL категорий
        for category, keywords in self.ymyl_categories.items():
            score = sum(content_lower.count(kw) for kw in keywords)
            category_scores[category] = score
        
        # Подсчет упоминаний ключевых слов для не-YMYL категорий
        for category, keywords in self.non_ymyl_categories.items():
            score = sum(content_lower.count(kw) for kw in keywords)
            category_scores[category] = score
        
        # Определение категории с наивысшим счетом
        if not category_scores:
            return 'unknown', False
            
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        # Если счет слишком низкий, считаем категорию неопределенной
        if best_category[1] < 3:
            return 'unknown', False
            
        is_ymyl = best_category[0] in self.ymyl_categories
        return best_category[0], is_ymyl
    
    def fetch_content(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Получение текстового содержимого страницы
        
        Args:
            url: URL страницы
            
        Returns:
            Кортеж (html_content, text_content)
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаление скриптов, стилей и других ненужных элементов
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Получение основного контента
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if main_content:
                # Если найден основной контент, используем его
                text_content = main_content.get_text(separator='\n\n')
            else:
                # Иначе используем весь текст страницы
                text_content = soup.get_text(separator='\n\n')
            
            # Очистка текста от лишних пробелов и пустых строк
            text_content = re.sub(r'\n{3,}', '\n\n', text_content)
            text_content = re.sub(r'\s{2,}', ' ', text_content)
            
            return html_content, text_content.strip()
            
        except Exception as e:
            logger.error(f"Ошибка при получении контента с {url}: {e}")
            return None, None
    
    def analyze_content(self, url: str, text_content: str, category: str, is_ymyl: bool) -> Dict:
        """
        Анализ содержимого и автоматическая разметка E-E-A-T метрик
        
        Args:
            url: URL страницы
            text_content: Текстовое содержимое
            category: Категория контента
            is_ymyl: YMYL-статус
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            # Анализ с помощью EnhancedEEATAnalyzer
            eeat_analysis = self.analyzer.analyze(text_content, industry=category)
            
            # Создание записи данных с разметкой
            data_entry = {
                'url': url,
                'category': category,
                'is_ymyl': is_ymyl,
                'content_sample': text_content[:1000] + '...' if len(text_content) > 1000 else text_content,
                'content_length': len(text_content.split()),
                'expertise_score': eeat_analysis['expertise_score'],
                'authority_score': eeat_analysis['authority_score'],
                'trust_score': eeat_analysis['trust_score'],
                'structural_score': eeat_analysis['structural_score'],
                'semantic_coherence_score': eeat_analysis['semantic_coherence_score'],
                'citation_score': eeat_analysis.get('citation_score', 0.0),
                'external_links_score': eeat_analysis.get('external_links_score', 0.0),
                'overall_eeat_score': eeat_analysis['overall_eeat_score'],
                'auto_analyzed': True,
                'expert_validated': False
            }
            
            return data_entry
            
        except Exception as e:
            logger.error(f"Ошибка при анализе контента с {url}: {e}")
            return None
    
    def collect_from_url_list(self, url_list: List[str], max_workers: int = 5) -> List[Dict]:
        """
        Сбор и анализ контента с заданного списка URL
        
        Args:
            url_list: Список URL для анализа
            max_workers: Максимальное количество параллельных потоков
            
        Returns:
            Список словарей с данными
        """
        collected_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Определяем функцию обработки одного URL
            def process_url(url):
                try:
                    # Получаем контент
                    html_content, text_content = self.fetch_content(url)
                    
                    if not text_content or len(text_content.split()) < 200:
                        logger.warning(f"Недостаточно контента на {url}")
                        return None
                    
                    # Определяем категорию
                    category, is_ymyl = self.identify_category(url, text_content)
                    
                    # Анализируем контент
                    data_entry = self.analyze_content(url, text_content, category, is_ymyl)
                    
                    if data_entry:
                        return data_entry
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке {url}: {e}")
                    return None
            
            # Обработка всех URL с отображением прогресса
            results = list(tqdm(
                executor.map(process_url, url_list),
                total=len(url_list),
                desc="Collecting content"
            ))
        
        # Фильтрация None результатов
        collected_data = [data for data in results if data]
        
        # Сохранение результатов
        self._save_collected_data(collected_data)
        
        return collected_data
    
    def collect_from_search_queries(
        self, 
        queries: List[str], 
        results_per_query: int = 10,
        api_key: Optional[str] = None
    ) -> List[Dict]:
        """
        Сбор и анализ контента из результатов поиска
        
        Примечание: Требуется API ключ для поисковой системы.
        В данном примере код не реализован полностью, так как требует API ключа.
        
        Args:
            queries: Список поисковых запросов
            results_per_query: Количество результатов на запрос
            api_key: API ключ для поисковой системы
            
        Returns:
            Список словарей с данными
        """
        collected_urls = []
        
        # В реальной имплементации здесь был бы код для запросов к поисковой системе
        # и извлечения URL результатов. Поскольку это требует API ключа, 
        # мы просто добавим заглушку.
        
        if not api_key:
            logger.warning("API ключ не предоставлен, поиск не может быть выполнен")
            return []
        
        # Заглушка для демонстрации
        for query in queries:
            logger.info(f"Поиск по запросу: {query}")
            # Здесь был бы запрос к API поисковой системы
            # И обработка результатов
            
            # Добавляем случайные URL для демонстрации
            query_urls = [
                f"https://example.com/result-{query.replace(' ', '-')}-{i}"
                for i in range(results_per_query)
            ]
            collected_urls.extend(query_urls)
        
        # Обработка собранных URL
        return self.collect_from_url_list(collected_urls)
    
    def _save_collected_data(self, data: List[Dict]) -> None:
        """
        Сохранение собранных данных
        
        Args:
            data: Список словарей с данными
        """
        if not data:
            logger.warning("Нет данных для сохранения")
            return
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Сохранение в CSV
        csv_path = os.path.join(self.output_dir, f"eeat_data_{timestamp}.csv")
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        # Сохранение в JSON
        json_path = os.path.join(self.output_dir, f"eeat_data_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Сохранено {len(data)} записей в {csv_path} и {json_path}")
    
    def generate_validation_form(self, data: List[Dict], output_file: str) -> None:
        """
        Генерация формы для экспертной валидации
        
        Args:
            data: Список словарей с данными
            output_file: Путь к файлу для сохранения формы
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>E-E-A-T Expert Validation Form</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .item { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; }
                .content { background: #f9f9f9; padding: 10px; margin: 10px 0; max-height: 300px; overflow-y: auto; }
                .metrics { display: flex; flex-wrap: wrap; }
                .metric { width: 25%; margin-bottom: 10px; }
                .metric input { width: 60px; }
                .buttons { margin-top: 20px; }
                button { padding: 10px 20px; margin-right: 10px; }
            </style>
        </head>
        <body>
            <h1>E-E-A-T Expert Validation Form</h1>
            <p>Please review the content and adjust the E-E-A-T metrics as needed.</p>
            
            <form id="validation-form">
        """
        
        # Добавление элементов формы для каждого элемента данных
        for i, item in enumerate(data):
            html += f"""
            <div class="item">
                <h3>Item #{i+1}: {item['category'].title()} {'(YMYL)' if item['is_ymyl'] else ''}</h3>
                <p>URL: <a href="{item['url']}" target="_blank">{item['url']}</a></p>
                
                <div class="content">{item['content_sample']}</div>
                
                <div class="metrics">
                    <div class="metric">
                        <label for="expertise_{i}">Expertise (E):</label>
                        <input type="number" id="expertise_{i}" name="expertise_{i}" min="0" max="1" step="0.01" value="{item['expertise_score']:.2f}">
                    </div>
                    <div class="metric">
                        <label for="authority_{i}">Authority (A):</label>
                        <input type="number" id="authority_{i}" name="authority_{i}" min="0" max="1" step="0.01" value="{item['authority_score']:.2f}">
                    </div>
                    <div class="metric">
                        <label for="trust_{i}">Trust (T):</label>
                        <input type="number" id="trust_{i}" name="trust_{i}" min="0" max="1" step="0.01" value="{item['trust_score']:.2f}">
                    </div>
                    <div class="metric">
                        <label for="structure_{i}">Structure:</label>
                        <input type="number" id="structure_{i}" name="structure_{i}" min="0" max="1" step="0.01" value="{item['structural_score']:.2f}">
                    </div>
                    <div class="metric">
                        <label for="semantic_{i}">Semantic:</label>
                        <input type="number" id="semantic_{i}" name="semantic_{i}" min="0" max="1" step="0.01" value="{item['semantic_coherence_score']:.2f}">
                    </div>
                    <div class="metric">
                        <label for="citation_{i}">Citation:</label>
                        <input type="number" id="citation_{i}" name="citation_{i}" min="0" max="1" step="0.01" value="{item['citation_score']:.2f}">
                    </div>
                    <div class="metric">
                        <label for="links_{i}">External Links:</label>
                        <input type="number" id="links_{i}" name="links_{i}" min="0" max="1" step="0.01" value="{item['external_links_score']:.2f}">
                    </div>
                    <div class="metric">
                        <label for="overall_{i}">Overall E-E-A-T:</label>
                        <input type="number" id="overall_{i}" name="overall_{i}" min="0" max="1" step="0.01" value="{item['overall_eeat_score']:.2f}">
                    </div>
                </div>
                
                <div class="category">
                    <label for="category_{i}">Category:</label>
                    <select id="category_{i}" name="category_{i}">
            """
            
            # Добавление опций для категорий
            categories = list(self.ymyl_categories.keys()) + list(self.non_ymyl_categories.keys()) + ['unknown']
            for cat in categories:
                selected = 'selected' if cat == item['category'] else ''
                html += f'<option value="{cat}" {selected}>{cat.title()}</option>'
            
            html += f"""
                    </select>
                    <label for="is_ymyl_{i}">YMYL:</label>
                    <input type="checkbox" id="is_ymyl_{i}" name="is_ymyl_{i}" {"checked" if item['is_ymyl'] else ""}>
                </div>
                
                <input type="hidden" name="url_{i}" value="{item['url']}">
                <input type="hidden" name="content_sample_{i}" value="{item['content_sample'].replace('"', '&quot;')}">
                <input type="hidden" name="content_length_{i}" value="{item['content_length']}">
            </div>
            """
        
        html += """
            <div class="buttons">
                <button type="submit">Save Validation</button>
                <button type="reset">Reset</button>
            </div>
            </form>
            
            <script>
                document.getElementById('validation-form').addEventListener('submit', function(e) {
                    e.preventDefault();
                    const formData = new FormData(this);
                    const results = [];
                    
                    // Количество элементов
                    const itemCount = """ + str(len(data)) + """;
                    
                    // Собираем данные
                    for (let i = 0; i < itemCount; i++) {
                        results.push({
                            url: formData.get(`url_${i}`),
                            category: formData.get(`category_${i}`),
                            is_ymyl: formData.get(`is_ymyl_${i}`) === 'on',
                            content_sample: formData.get(`content_sample_${i}`),
                            content_length: parseInt(formData.get(`content_length_${i}`)),
                            expertise_score: parseFloat(formData.get(`expertise_${i}`)),
                            authority_score: parseFloat(formData.get(`authority_${i}`)),
                            trust_score: parseFloat(formData.get(`trust_${i}`)),
                            structural_score: parseFloat(formData.get(`structure_${i}`)),
                            semantic_coherence_score: parseFloat(formData.get(`semantic_${i}`)),
                            citation_score: parseFloat(formData.get(`citation_${i}`)),
                            external_links_score: parseFloat(formData.get(`links_${i}`)),
                            overall_eeat_score: parseFloat(formData.get(`overall_${i}`)),
                            auto_analyzed: true,
                            expert_validated: true
                        });
                    }
                    
                    // Сохранение результатов
                    const blob = new Blob([JSON.stringify(results, null, 2)], {type: 'application/json'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'eeat_validated_data.json';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                });
            </script>
        </body>
        </html>
        """
        
        # Сохранение формы
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Форма для валидации сохранена в {output_file}")

# Пример использования
if __name__ == "__main__":
    # Список URL-адресов известных авторитетных сайтов по категориям
    urls = {
        'finance': [
            'https://www.investopedia.com/terms/i/investment.asp',
            'https://www.bloomberg.com/markets',
            'https://www.cnbc.com/personal-finance/'
        ],
        'health': [
            'https://www.mayoclinic.org/healthy-lifestyle',
            'https://www.webmd.com/diet/features/the-truth-about-eating-for-health',
            'https://www.nih.gov/health-information'
        ],
        'tech': [
            'https://www.wired.com/tag/artificial-intelligence/',
            'https://techcrunch.com/',
            'https://www.theverge.com/tech'
        ],
        'travel': [
            'https://www.lonelyplanet.com/articles/best-in-travel',
            'https://www.tripadvisor.com/TravelersChoice',
            'https://www.booking.com/destinationfinder.html'
        ]
    }
    
    # Собираем все URL в один список
    all_urls = [url for category_urls in urls.values() for url in category_urls]
    
    # Путь к модели E-E-A-T
    model_path = '/content/seo-ai-models/models/checkpoints/eeat_best_model.joblib'
    
    # Инициализация коллектора
    collector = WebContentCollector(output_dir="real_eeat_data", model_path=model_path)
    
    # Сбор данных
    print(f"Collecting data from {len(all_urls)} URLs...")
    collected_data = collector.collect_from_url_list(all_urls)
    
    # Генерация формы для экспертной валидации
    if collected_data:
        collector.generate_validation_form(collected_data, "eeat_validation_form.html")
        print(f"Collected {len(collected_data)} items. Validation form generated.")
