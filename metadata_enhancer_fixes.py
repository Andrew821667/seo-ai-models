def improved_date_extraction(self, published_date, gost, url):
    if published_date:
        try:
            # Попытка найти год в формате 4 цифр
            year = re.search(r'\d{4}', published_date).group(0)
            gost += f"— {year}. "
        except AttributeError:
            # Если не удалось найти 4 цифры, пробуем другие форматы
            self.logger.warning(f"Не удалось извлечь год из даты: {published_date}")
            
            # Проверка на наличие 2 цифр, которые могут быть годом (например, '22 для 2022)
            year_match = re.search(r'\b\d{2}\b', published_date)
            if year_match:
                year_short = year_match.group(0)
                current_century = datetime.now().year // 100
                year = f"{current_century}{year_short}"
                self.logger.info(f"Извлечен короткий год: {year_short}, преобразован в: {year}")
                gost += f"— {year}. "
            
            # Проверка на текстовое представление даты
            months_ru = ['январ', 'феврал', 'март', 'апрел', 'ма[йя]', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр']
            months_en = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
            
            month_pattern_ru = '|'.join(months_ru)
            month_pattern_en = '|'.join(months_en)
            
            date_match_ru = re.search(fr'({month_pattern_ru}).*?(\d{{4}})', published_date, re.IGNORECASE)
            date_match_en = re.search(fr'({month_pattern_en}).*?(\d{{4}})', published_date, re.IGNORECASE)
            
            if date_match_ru:
                year = date_match_ru.group(2)
                self.logger.info(f"Извлечен год из русской текстовой даты: {year}")
                gost += f"— {year}. "
            elif date_match_en:
                year = date_match_en.group(2)
                self.logger.info(f"Извлечен год из английской текстовой даты: {year}")
                gost += f"— {year}. "
            else:
                # Если ничего не найдено, добавляем информацию о доступе
                accessed_date = datetime.now().strftime("%d.%m.%Y")
                gost += f"— (дата обращения: {accessed_date}). "
    
    if url:
        gost += f"URL: {url}"
    
    return gost.strip()
