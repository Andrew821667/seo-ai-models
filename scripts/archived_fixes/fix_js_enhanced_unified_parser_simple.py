# Путь к файлу
file_path = "seo_ai_models/parsers/unified/js_enhanced_unified_parser.py"

# Чтение файла
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Найдем TODO комментарий
todo_comment = "# TODO: Извлечь больше информации из HTML"

if todo_comment in content:
    # Заменим TODO на реальный код
    html_extraction_code = """        # Извлекаем дополнительную информацию из HTML
        
        # 1. Извлекаем информацию о формах
        forms_info = []
        forms = soup.find_all('form')
        for i, form in enumerate(forms):
            form_info = {
                'id': i,
                'action': form.get('action', ''),
                'method': form.get('method', 'get').upper(),
                'fields': []
            }
            
            # Извлекаем информацию о полях формы
            inputs = form.find_all(['input', 'select', 'textarea'])
            for input_field in inputs:
                field_info = {
                    'type': input_field.name,
                    'name': input_field.get('name', ''),
                    'id': input_field.get('id', ''),
                    'required': input_field.has_attr('required'),
                }
                
                if input_field.name == 'input':
                    field_info['input_type'] = input_field.get('type', 'text')
                
                form_info['fields'].append(field_info)
            
            forms_info.append(form_info)
        
        if forms_info:
            result['forms'] = forms_info
        
        # 2. Извлекаем информацию о медиа-контенте
        media_info = {
            'images': [],
            'videos': [],
            'audio': []
        }
        
        # Изображения
        images = soup.find_all('img')
        for img in images:
            img_info = {
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'width': img.get('width', ''),
                'height': img.get('height', ''),
                'lazy_loading': img.get('loading') == 'lazy'
            }
            media_info['images'].append(img_info)
        
        # Видео
        videos = soup.find_all(['video', 'iframe'])
        for video in videos:
            if video.name == 'video':
                video_info = {
                    'src': video.get('src', ''),
                    'type': 'html5',
                    'width': video.get('width', ''),
                    'height': video.get('height', ''),
                    'controls': video.has_attr('controls'),
                    'autoplay': video.has_attr('autoplay')
                }
            else:  # iframe
                src = video.get('src', '')
                video_type = 'unknown'
                if 'youtube' in src:
                    video_type = 'youtube'
                elif 'vimeo' in src:
                    video_type = 'vimeo'
                
                video_info = {
                    'src': src,
                    'type': video_type,
                    'width': video.get('width', ''),
                    'height': video.get('height', '')
                }
            
            media_info['videos'].append(video_info)
        
        # Аудио
        audios = soup.find_all('audio')
        for audio in audios:
            audio_info = {
                'src': audio.get('src', ''),
                'controls': audio.has_attr('controls'),
                'autoplay': audio.has_attr('autoplay')
            }
            media_info['audio'].append(audio_info)
        
        if any(media_info.values()):
            result['media'] = media_info"""
    
    content = content.replace(todo_comment, html_extraction_code)
    
    # Записываем изменения обратно в файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"TODO комментарий в файле {file_path} успешно заменен на реальный код")
else:
    print(f"TODO комментарий не найден в файле {file_path}")
