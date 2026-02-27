"""
Digital Smarty v2.0 – Telegram Bot (Pyrogram)
AI Meeting Analyzer with Deepgram + GPT-4o
Supports files up to 2GB via Telegram MTProto
"""

import os
import re
import uuid
import asyncio
import logging
import tempfile

from pyrogram import Client, filters
from pyrogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery,
)

from pipeline import (
    download_from_url, transcribe_file, analyze_meeting,
    generate_pdf, generate_html, generate_txt, format_ts,
)

# -- Logging --
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("digital_smarty")

# -- Config --
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

TMP = tempfile.gettempdir()

# -- Flush old updates on startup --
def flush_old_updates():
    """Delete webhook and flush pending updates before Pyrogram starts."""
    import requests as _req
    base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    try:
        r = _req.get(f"{base}/deleteWebhook?drop_pending_updates=true", timeout=10)
        log.info(f"deleteWebhook: {r.json()}")
        r2 = _req.get(f"{base}/getUpdates?offset=-1&timeout=0", timeout=10)
        log.info(f"getUpdates flush: {r2.status_code}")
    except Exception as e:
        log.warning(f"Flush failed (non-critical): {e}")

flush_old_updates()

# -- Pyrogram Client (in-memory session) --
from pyrogram.session import Session
Session.notice_displayed = True  # suppress Pyrogram notice

app = Client(
    "digital_smarty_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=TELEGRAM_TOKEN,
    workdir=TMP,
    in_memory=True,  # no .session file needed
)

# -- Sessions --
sessions = {}

LANGUAGES = {
    "ru": ("🇷🇺 Русский", "ru"),
    "en": ("🇬🇧 English", "en"),
    "kz": ("🇰🇿 Қазақша", "kk"),
    "es": ("🇪🇸 Español", "es"),
    "zh": ("🇨🇳 中文", "zh"),
    "orig": ("🗣 Язык оригинала", "original"),
}

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".opus", ".flac", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS


def get_session(cid):
    if cid not in sessions:
        sessions[cid] = {"files": [], "urls": [], "processing": False}
    return sessions[cid]


def reset_session(cid):
    sessions[cid] = {"files": [], "urls": [], "processing": False}


# ═══════════════════════════════════════════════════
# FILE DOWNLOAD (Pyrogram – up to 2GB)
# ═══════════════════════════════════════════════════

async def download_tg_file(message_or_media, file_name):
    """Download file via Pyrogram MTProto (up to 2GB)."""
    path = os.path.join(TMP, f"ds_{uuid.uuid4().hex[:8]}_{file_name}")
    downloaded = await message_or_media.download(file_name=path)
    size = os.path.getsize(downloaded) if os.path.exists(downloaded) else 0
    log.info(f"Downloaded: {file_name} ({size} bytes) via MTProto")
    return downloaded


# ═══════════════════════════════════════════════════
# PROCESSING
# ═══════════════════════════════════════════════════

async def process_meeting(client, chat_id, lang_code):
    s = get_session(chat_id)
    try:
        s["processing"] = True
        n_files = len(s["files"]) + len(s["urls"])
        await client.send_message(
            chat_id,
            f"⏳ Принято! Запускаю анализ ({n_files} источник(ов))...\n\n"
            "1️⃣ Скачиваю и извлекаю аудио\n"
            "2️⃣ Транскрибирую (Deepgram Nova-2)\n"
            "3️⃣ Анализирую как эксперт (GPT-4o)\n"
            "4️⃣ Генерирую отчёты\n\n"
            "Это может занять несколько минут ☕",
        )

        all_transcripts = []

        # Process Telegram files (downloaded via MTProto)
        for fi in s["files"]:
            await client.send_message(chat_id, f"⬇️ Скачиваю: {fi['name']}...")
            # fi["msg"] is the original message object – download from it
            path = await download_tg_file(fi["msg"], fi["name"])
            await client.send_message(chat_id, "🎙 Транскрибирую...")
            t = await asyncio.to_thread(transcribe_file, path, DEEPGRAM_API_KEY)
            all_transcripts.append(t)

        # Process URLs
        for url in s["urls"]:
            await client.send_message(chat_id, "⬇️ Скачиваю по ссылке...")
            path = await asyncio.to_thread(download_from_url, url)
            await client.send_message(chat_id, "🎙 Транскрибирую...")
            t = await asyncio.to_thread(transcribe_file, path, DEEPGRAM_API_KEY)
            all_transcripts.append(t)

        if not all_transcripts:
            await client.send_message(chat_id, "😅 Не удалось обработать файлы. Попробуй ещё раз!")
            reset_session(chat_id)
            return

        # Merge
        if len(all_transcripts) == 1:
            merged = all_transcripts[0]
        else:
            merged = {
                "full_text": "\n\n".join(t["full_text"] for t in all_transcripts),
                "speaker_transcript": "\n\n--- (продолжение) ---\n\n".join(
                    t["speaker_transcript"] for t in all_transcripts
                ),
                "speakers_count": max(t["speakers_count"] for t in all_transcripts),
                "detected_language": all_transcripts[0]["detected_language"],
                "duration_seconds": sum(t["duration_seconds"] for t in all_transcripts),
            }

        await client.send_message(chat_id, "🧠 Анализирую содержание...")
        analysis = await asyncio.to_thread(analyze_meeting, merged, lang_code, OPENAI_API_KEY)

        await client.send_message(chat_id, "📝 Генерирую отчёты...")
        pdf_path, pdf_fn = await asyncio.to_thread(generate_pdf, analysis)
        html_path, html_fn = await asyncio.to_thread(generate_html, analysis)
        txt_path, txt_fn = await asyncio.to_thread(generate_txt, analysis, merged["speaker_transcript"])

        await client.send_message(
            chat_id,
            "✅ **Готово!**\n\n"
            "📄 PDF – структурированный отчёт\n"
            "🌐 HTML – интерактивный (открой в браузере)\n"
            "📝 TXT – полная транскрипция\n\n"
            "Есть ещё записи? Скидывай! 💪",
        )

        await client.send_document(chat_id, pdf_path, file_name=pdf_fn, caption="📄 PDF-отчёт")
        await client.send_document(chat_id, html_path, file_name=html_fn, caption="🌐 Интерактивный HTML")
        await client.send_document(chat_id, txt_path, file_name=txt_fn, caption="📝 Транскрипция")

        for p in [pdf_path, html_path, txt_path]:
            if os.path.exists(p):
                os.remove(p)

    except Exception as ex:
        log.error(f"Error: {ex}", exc_info=True)
        await client.send_message(chat_id, f"😅 Ошибка: {str(ex)[:400]}\n\nПопробуй ещё раз!")
    finally:
        reset_session(chat_id)


# ═══════════════════════════════════════════════════
# HANDLERS
# ═══════════════════════════════════════════════════

@app.on_message(filters.command(["start", "help"]))
async def handle_start(client, message: Message):
    await message.reply(
        "👋 Привет! Я **Цифровой Умник** – AI-аналитик встреч.\n\n"
        "Закидывай аудио/видео или ссылку YouTube/Google Drive.\n"
        "Когда всё загружено – жми /analyze\n\n"
        "🎙 **Принимаю:**\n"
        "• Аудио: mp3, wav, ogg, m4a, opus, flac\n"
        "• Видео: mp4, mov, avi, mkv, webm\n"
        "• Голосовые и видеосообщения\n"
        "• Ссылки YouTube / Google Drive\n"
        "• Файлы до 2 ГБ 💪\n\n"
        "🚀 Готов к работе!",
    )


@app.on_message(filters.command("analyze"))
async def handle_analyze(client, message: Message):
    s = get_session(message.chat.id)
    if s["processing"]:
        await message.reply("⏳ Ещё обрабатываю. Подожди!")
        return
    if not s["files"] and not s["urls"]:
        await message.reply("🤔 Файлов нет! Сначала скинь аудио/видео или ссылку.")
        return
    buttons = []
    for code, (name, _) in LANGUAGES.items():
        buttons.append([InlineKeyboardButton(name, callback_data=f"lang_{code}")])
    await message.reply(
        "🌍 На каком языке написать отчёт?",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


@app.on_callback_query(filters.regex(r"^lang_"))
async def handle_language(client, callback: CallbackQuery):
    s = get_session(callback.message.chat.id)
    if s["processing"]:
        await callback.answer("Уже обрабатываю!")
        return
    key = callback.data[5:]
    name, lang_code = LANGUAGES.get(key, ("", "ru"))
    await callback.answer(f"Выбран: {name}")
    await callback.message.edit_text(f"🌍 Язык отчёта: {name}")
    asyncio.create_task(process_meeting(client, callback.message.chat.id, lang_code))


@app.on_message(filters.audio | filters.voice)
async def handle_audio(client, message: Message):
    s = get_session(message.chat.id)
    if s["processing"]:
        return
    if message.audio:
        fn = message.audio.file_name or f"audio_{uuid.uuid4().hex[:6]}.mp3"
    else:
        fn = f"voice_{uuid.uuid4().hex[:6]}.ogg"
    # Store the message itself for Pyrogram download
    s["files"].append({"msg": message, "name": fn})
    await message.reply(f"📎 Принято: **{fn}**\nЕщё? Или /analyze")


@app.on_message(filters.video | filters.video_note)
async def handle_video(client, message: Message):
    s = get_session(message.chat.id)
    if s["processing"]:
        return
    if message.video:
        fn = message.video.file_name or f"video_{uuid.uuid4().hex[:6]}.mp4"
    else:
        fn = f"videonote_{uuid.uuid4().hex[:6]}.mp4"
    s["files"].append({"msg": message, "name": fn})
    await message.reply(f"📎 Принято: **{fn}**\nЕщё? Или /analyze")


@app.on_message(filters.document)
async def handle_document(client, message: Message):
    s = get_session(message.chat.id)
    if s["processing"]:
        return
    fn = message.document.file_name or "file"
    ext = os.path.splitext(fn)[1].lower()
    if ext in MEDIA_EXTS:
        s["files"].append({"msg": message, "name": fn})
        await message.reply(f"📎 Принято: **{fn}**\nЕщё? Или /analyze")
    else:
        await message.reply(
            f"🤔 **{fn}** – не аудио/видео.\nПоддерживаю: mp3, wav, m4a, mp4, mov..."
        )


@app.on_message(filters.text & ~filters.command(["start", "help", "analyze"]))
async def handle_text(client, message: Message):
    s = get_session(message.chat.id)
    if s["processing"]:
        return
    text = message.text.strip()

    url_patterns = [
        r"https?://(?:www\.)?youtube\.com/\S+",
        r"https?://youtu\.be/\S+",
        r"https?://drive\.google\.com/\S+",
        r"https?://\S+\.(?:mp3|wav|ogg|m4a|mp4|mov|avi|mkv|webm)",
    ]
    for pattern in url_patterns:
        match = re.search(pattern, text)
        if match:
            s["urls"].append(match.group(0))
            await message.reply("🔗 Ссылка принята!\nЕщё? Или /analyze")
            return

    if re.match(r"https?://\S+", text):
        s["urls"].append(text)
        await message.reply("🔗 Ссылка принята!\nЕщё? Или /analyze")
        return

    await message.reply(
        "👋 Скинь мне аудио/видео или ссылку YouTube/Google Drive.\n"
        "Когда всё загружено – жми /analyze 🚀",
    )


# ═══════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("🧠 Digital Smarty v2.0 starting (Pyrogram MTProto)...")
    log.info(f"API_ID={API_ID}, TMP={TMP}")
    log.info("Flushed old updates. Starting polling...")
    app.run()
