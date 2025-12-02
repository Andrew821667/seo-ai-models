#!/bin/bash
# Скрипт для скачивания артефактов из GitHub Actions

REPO="Andrew821667/seo-ai-models"
RUN_ID="19866025936"
ARTIFACT_IDS=("4740235401" "4740235795")
ARTIFACT_NAMES=("seo-analysis-results-48" "recommendations-48")

echo "Для скачивания артефактов нужен GitHub Personal Access Token"
echo "Создайте токен здесь: https://github.com/settings/tokens"
echo "Требуемые разрешения: repo (полный доступ к репозиториям)"
echo ""
read -p "Введите GitHub Token: " GITHUB_TOKEN
echo ""

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Токен не предоставлен"
    exit 1
fi

for i in "${!ARTIFACT_IDS[@]}"; do
    ARTIFACT_ID="${ARTIFACT_IDS[$i]}"
    ARTIFACT_NAME="${ARTIFACT_NAMES[$i]}"

    echo "📥 Скачивание: $ARTIFACT_NAME..."

    curl -L \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/$REPO/actions/artifacts/$ARTIFACT_ID/zip" \
        -o "${ARTIFACT_NAME}.zip"

    if [ $? -eq 0 ]; then
        echo "✅ Скачано: ${ARTIFACT_NAME}.zip"
        echo "📦 Распаковка..."
        unzip -o "${ARTIFACT_NAME}.zip"
        rm "${ARTIFACT_NAME}.zip"
        echo "✅ Распаковано"
    else
        echo "❌ Ошибка при скачивании $ARTIFACT_NAME"
    fi
    echo ""
done

echo "🎉 Готово!"
echo ""
echo "Скачанные файлы:"
ls -lh analysis_result.json recommendations_ru.md 2>/dev/null || echo "Файлы не найдены"
