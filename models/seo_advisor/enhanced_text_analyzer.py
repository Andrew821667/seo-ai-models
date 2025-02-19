
# [Предыдущий код до метода _analyze_structure остается таким же]

    def _analyze_structure(self, content: str) -> Dict:
        """Улучшенный анализ структуры текста"""
        # Разбиваем на строки, сохраняя структуру
        lines = content.strip().split('\n')
        
        # Улучшенное определение заголовков
        headers = []
        current_header = None
        header_pattern = re.compile(r'^(?:#{1,6}\s|[A-ZА-Я][^.!?]*(?::|$)|.*?\n[-=]+$)')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Проверяем маркеры заголовков
            if header_pattern.match(line):
                # Определяем уровень заголовка
                level = 1
                if line.startswith('#'):
                    level = len(re.match(r'^#+', line).group())
                    line = line.lstrip('#').strip()
                elif i + 1 < len(lines) and re.match(r'^[-=]+$', lines[i + 1]):
                    level = 1 if '=' in lines[i + 1] else 2
                elif line.isupper() or line.endswith(':'):
                    level = 2
                
                # Проверяем, что это не элемент списка
                if not re.match(r'^[-*•]\s|^\d+\.\s', line):
                    headers.append({
                        'text': line.rstrip(':'),
                        'level': level
                    })
                    current_header = line
        
        # Улучшенное определение списков
        lists = []
        current_list = []
        list_patterns = [
            (r'^\s*[-*•]\s+(.+)$', 'unordered'),
            (r'^\s*\d+\.\s+(.+)$', 'ordered')
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_list:
                    if len(current_list) > 1:  # Игнорируем одиночные элементы
                        lists.append(current_list)
                    current_list = []
                continue
            
            is_list_item = False
            for pattern, list_type in list_patterns:
                match = re.match(pattern, line)
                if match:
                    item_text = match.group(1).strip()
                    # Проверяем, что это не разделитель
                    if not re.match(r'^[-=]+$', item_text):
                        current_list.append(item_text)
                        is_list_item = True
                        break
            
            if not is_list_item and current_list:
                if len(current_list) > 1:
                    lists.append(current_list)
                current_list = []
        
        if current_list and len(current_list) > 1:
            lists.append(current_list)
        
        # Улучшенное определение изображений
        images = []
        image_patterns = [
            r'\[Изображение:\s*([^\]]+)\]',  # Markdown-подобный синтаксис
            r'<img[^>]+alt=["\'](.*?)["\'][^>]*>',  # HTML
            r'!\[([^\]]*)\]\([^)]+\)'  # Markdown
        ]
        
        for pattern in image_patterns:
            for match in re.finditer(pattern, content):
                alt_text = match.group(1).strip()
                if alt_text:
                    images.append({'alt': alt_text, 'src': ''})
        
        # Улучшенное определение параграфов и предложений
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            # Пропускаем заголовки и элементы списков
            if not any(pattern.match(line) for pattern, _ in list_patterns) and \
               not header_pattern.match(line):
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Разбиваем на предложения с учетом сокращений
        sentences = []
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-Я])')
        
        for paragraph in paragraphs:
            for sent in sentence_pattern.split(paragraph):
                sent = sent.strip()
                if sent and not re.match(r'^[-=]+$', sent):
                    sentences.append(sent)
        
        # Подсчет слов и символов
        words = [w for w in re.findall(r'\b[\w\'-]+\b', content.lower()) 
                if not w.isnumeric()]
        word_count = len(words)
        char_count = sum(len(word) for word in words)
        
        # Поиск ссылок
        links = re.findall(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            content
        )
        
        return {
            'paragraphs': paragraphs,
            'sentences': sentences,
            'headers': headers,
            'lists': lists,
            'links': links,
            'images': images,
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': char_count / word_count if word_count > 0 else 0,
            'avg_sentence_length': word_count / len(sentences) if sentences else 0
        }
