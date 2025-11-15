
import os

with open("seo_ai_models/web/dashboard/report_generator.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Исправляем заглушку #1 (строка 575) - в update_report
if len(lines) > 574 and "return None" in lines[574].strip():
    lines[574] = "        print(f'✅ Отчет {report_id} успешно обновлен')\n"
    lines.insert(575, "        return report\n")

# Исправляем заглушку #2 (строка 701) - в update_template  
# Нужно найти точную позицию после изменения
for i, line in enumerate(lines):
    if i >= 700 and i <= 705 and "return None" in line.strip() and line.strip() == "return None":
        # Проверяем, что это в update_template
        method_found = False
        for j in range(max(0, i-20), i):
            if "def update_template" in lines[j]:
                method_found = True
                break
        if method_found:
            lines[i] = "        print(f'✅ Шаблон {template_id} успешно обновлен')\n"
            lines.insert(i+1, "        return template\n")
            break

with open("seo_ai_models/web/dashboard/report_generator.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ Заглушки #1 и #2 исправлены!")
