
from typing import List, Dict
import re

class ImageAnalyzer:
    def analyze_images(self, content: str) -> List[Dict[str, str]]:
        """Улучшенный анализ изображений"""
        images = []
        
        # Поиск изображений в формате [Изображение: текст]
        markdown_pattern = r'\[Изображение:\s*([^\]]+)\]'
        for match in re.finditer(markdown_pattern, content):
            alt_text = match.group(1).strip()
            if alt_text:
                images.append({
                    'alt': alt_text,
                    'src': '',
                    'type': 'markdown'
                })
        
        # Поиск HTML изображений
        html_pattern = r'<img[^>]+src=["\'](.*?)["\'][^>]*alt=["\'](.*?)["\'][^>]*>'
        for match in re.finditer(html_pattern, content):
            src = match.group(1)
            alt = match.group(2)
            if src or alt:
                images.append({
                    'src': src,
                    'alt': alt,
                    'type': 'html'
                })
        
        # Поиск изображений в формате Markdown ![alt](url)
        md_image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(md_image_pattern, content):
            alt = match.group(1)
            src = match.group(2)
            if src:
                images.append({
                    'alt': alt,
                    'src': src,
                    'type': 'markdown_standard'
                })
        
        return images
