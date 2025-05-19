def improved_date_processing(self, prop, value, result):
    try:
        # Попытка разбора различных форматов даты
        parsed_date = dateutil.parser.parse(value)
        result[prop] = parsed_date.strftime('%Y-%m-%d')
    except Exception as e:
        self.logger.warning(f"Не удалось преобразовать дату '{value}' для свойства {prop}: {str(e)}")
        
        # Попытка извлечь дату с использованием регулярных выражений
        # Форматы: YYYY-MM-DD, DD.MM.YYYY, MM/DD/YYYY, и т.д.
        date_patterns = [
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD или YYYY/MM/DD
            r'(\d{1,2})[/.](\d{1,2})[/.](\d{4})',  # DD.MM.YYYY или DD/MM/YYYY
            r'(\d{4})(\d{2})(\d{2})'               # YYYYMMDD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, value)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if len(groups[0]) == 4:  # Первая группа - год (YYYY-MM-DD)
                        year, month, day = groups
                    elif len(groups[2]) == 4:  # Последняя группа - год (DD.MM.YYYY)
                        day, month, year = groups
                    
                    try:
                        # Проверяем, что значения в допустимом диапазоне
                        year_int = int(year)
                        month_int = int(month)
                        day_int = int(day)
                        
                        if 1 <= month_int <= 12 and 1 <= day_int <= 31 and 1900 <= year_int <= 2100:
                            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            result[prop] = formatted_date
                            self.logger.info(f"Дата успешно преобразована альтернативным способом: {value} -> {formatted_date}")
                            return
                    except ValueError:
                        continue
        
        # Если не удалось разобрать дату, оставляем оригинальное значение
        self.logger.info(f"Оставляем оригинальное значение даты: {value}")
        result[prop] = value
