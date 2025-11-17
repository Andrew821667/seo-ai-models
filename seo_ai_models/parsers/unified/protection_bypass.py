"""
Методы обхода защиты от парсинга для проекта SEO AI Models.
"""

import logging
import random
import json
import time
import re
from typing import Dict, List, Set, Optional, Any, Tuple


class BrowserFingerprint:
    """Генератор отпечатков браузера для обхода защиты от парсинга"""

    # Распространенные User-Agent для эмуляции реальных браузеров
    USER_AGENTS = [
        # Chrome на Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36",
        # Firefox на Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0",
        # Chrome на macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        # Safari на macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15",
        # Edge на Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 Edg/100.0.1185.50",
    ]

    # Языки и локали
    LANGUAGES = [
        "en-US,en;q=0.9",
        "en-GB,en;q=0.9",
        "fr-FR,fr;q=0.9,en;q=0.8",
        "de-DE,de;q=0.9,en;q=0.8",
        "ru-RU,ru;q=0.9,en;q=0.8",
    ]

    # Разрешения экранов
    SCREEN_RESOLUTIONS = [
        {"width": 1366, "height": 768},
        {"width": 1920, "height": 1080},
        {"width": 1536, "height": 864},
        {"width": 1440, "height": 900},
        {"width": 1280, "height": 720},
    ]

    def __init__(self, consistent: bool = True, seed: Optional[int] = None):
        """
        Инициализация генератора отпечатков браузера.

        Args:
            consistent: Генерировать ли одинаковый отпечаток при каждом вызове
            seed: Сид для генератора случайных чисел
        """
        self.consistent = consistent
        self.random = random.Random(seed if seed is not None else int(time.time()))

        if consistent:
            self.selected_user_agent = self.random.choice(self.USER_AGENTS)
            self.selected_language = self.random.choice(self.LANGUAGES)
            self.selected_resolution = self.random.choice(self.SCREEN_RESOLUTIONS)

    def get_user_agent(self) -> str:
        """Возвращает случайный или заранее выбранный User-Agent"""
        if self.consistent:
            return self.selected_user_agent
        return self.random.choice(self.USER_AGENTS)

    def get_accept_language(self) -> str:
        """Возвращает заголовок Accept-Language"""
        if self.consistent:
            return self.selected_language
        return self.random.choice(self.LANGUAGES)

    def get_screen_resolution(self) -> Dict[str, int]:
        """Возвращает разрешение экрана"""
        if self.consistent:
            return self.selected_resolution
        return self.random.choice(self.SCREEN_RESOLUTIONS)

    def get_all_headers(self) -> Dict[str, str]:
        """Возвращает все HTTP-заголовки для эмуляции реального браузера"""
        user_agent = self.get_user_agent()

        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": self.get_accept_language(),
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }

        return headers

    def get_browser_fingerprint_script(self) -> str:
        """
        Возвращает JavaScript-скрипт для подмены fingerprint браузера.

        Returns:
            str: JavaScript-код
        """
        screen_props = self.get_screen_resolution()

        js_template = """
        // Переопределяем свойства navigator
        try {
            Object.defineProperty(navigator, 'webdriver', {
                get: function() { return false; }
            });
        } catch (e) {}
        
        // Переопределяем свойства screen
        try {
            Object.defineProperty(screen, 'width', { get: function() { return {width}; } });
            Object.defineProperty(screen, 'height', { get: function() { return {height}; } });
            Object.defineProperty(screen, 'availWidth', { get: function() { return {width}; } });
            Object.defineProperty(screen, 'availHeight', { get: function() { return {height}; } });
        } catch (e) {}
        
        // Модифицируем Canvas для уникальности отпечатка
        const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
        CanvasRenderingContext2D.prototype.getImageData = function(sx, sy, sw, sh) {
            const imageData = originalGetImageData.call(this, sx, sy, sw, sh);
            const w = Math.floor(Math.random() * sw);
            const h = Math.floor(Math.random() * sh);
            const index = (h * sw + w) * 4;
            if (index < imageData.data.length - 4) {
                imageData.data[index] = (imageData.data[index] + 1) % 256;
            }
            return imageData;
        };
        """

        # Заменяем плейсхолдеры
        js_code = js_template.replace("{width}", str(screen_props["width"])).replace(
            "{height}", str(screen_props["height"])
        )

        return js_code
