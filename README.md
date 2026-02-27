# Digital Smarty v2.0

AI-аналитик встреч. Принимает аудио/видео файлы до 2 ГБ, транскрибирует и генерирует структурированные отчёты.

## Стек

- **Pyrogram** (MTProto) – приём файлов до 2 ГБ
- **Deepgram Nova-2** – транскрипция
- **GPT-4o** – анализ и генерация отчётов
- **ReportLab** – PDF-отчёты

## Переменные окружения

| Переменная | Описание |
|---|---|
| `TELEGRAM_TOKEN` | Токен бота от @BotFather |
| `TELEGRAM_API_ID` | API ID из https://my.telegram.org |
| `TELEGRAM_API_HASH` | API Hash из https://my.telegram.org |
| `OPENAI_API_KEY` | Ключ OpenAI API |
| `DEEPGRAM_API_KEY` | Ключ Deepgram API |

## Деплой на Railway

1. Форк репозитория
2. Создать проект в Railway, подключить GitHub-репо
3. Добавить все 5 переменных окружения
4. Railway автоматически соберёт и запустит бот

## Команды бота

- `/start` – приветствие и инструкция
- `/analyze` – запуск анализа загруженных файлов
- Отправить аудио/видео или ссылку YouTube/Google Drive
