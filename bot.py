"""
Digital Smarty v2.0 – Telegram Bot
AI Meeting Analyzer with Deepgram + GPT-4o
"""

import telebot
from telebot import types
import os
import re
import uuid
import threading
import logging
import tempfile

from pipeline import (
    download_from_url, transcribe_file, analyze_meeting,
    generate_pdf, generate_html, generate_txt, format_ts,
)

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("digital_smarty")

# ── Config ──
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

bot = telebot.TeleBot(TELEGRAM_TOKEN)
TMP = tempfile.gettempdir()

# ── Sessions ──
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
# PROCESSING
# ═══════════════════════════════════════════════════

def download_tg_file(file_id, file_name):
    info = bot.get_file(file_id)
    data = bot.download_file(info.file_path)
    path = os.path.join(TMP, f"ds_{uuid.uuid4().hex[:8]}_{file_name}")
    with open(path, "wb") as f:
        f.write(data)
    log.info(f"Downloaded: {file_name} ({len(data)} bytes)")
    return path


def process_meeting(chat_id, lang_code):
    s = get_session(chat_id)
    try:
        s["processing"] = True
        n_files = len(s["files"]) + len(s["urls"])
        bot.send_message(
            chat_id,
            f"⏳ Принято! Запускаю анализ ({n_files} источник(ов))...\n\n"
            "1️⃣ Скачиваю и извлекаю аудио\n"
            "2️⃣ Транскрибирую (Deepgram Nova-2)\n"
            "3️⃣ Анализирую как эксперт (GPT-4o)\n"
            "4️⃣ Генерирую отчёты\n\n"
            "Это может занять несколько минут ☕",
        )

        all_transcripts = []

        for fi in s["files"]:
            bot.send_message(chat_id, f"⬇️ Скачиваю: {fi['name']}...")
            path = download_tg_file(fi["id"], fi["name"])
            bot.send_message(chat_id, "🎙 Транскрибирую...")
            t = transcribe_file(path, DEEPGRAM_API_KEY)
            all_transcripts.append(t)

        for url in s["urls"]:
            bot.send_message(chat_id, "⬇️ Скачиваю по ссылке...")
            path = download_from_url(url)
            bot.send_message(chat_id, "🎙 Транскрибирую...")
            t = transcribe_file(path, DEEPGRAM_API_KEY)
            all_transcripts.append(t)

        if not all_transcripts:
            bot.send_message(chat_id, "😅 Не удалось обработать файлы. Попробуй ещё раз!")
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

        bot.send_message(chat_id, "🧠 Анализирую содержание...")
        analysis = analyze_meeting(merged, lang_code, OPENAI_API_KEY)

        bot.send_message(chat_id, "📝 Генерирую отчёты...")
        pdf_path, pdf_fn = generate_pdf(analysis)
        html_path, html_fn = generate_html(analysis)
        txt_path, txt_fn = generate_txt(analysis, merged["speaker_transcript"])

        bot.send_message(
            chat_id,
            "✅ *Готово!*\n\n"
            "📄 PDF – структурированный отчёт\n"
            "🌐 HTML – интерактивный (открой в браузере)\n"
            "📝 TXT – полная транскрипция\n\n"
            "Есть ещё записи? Скидывай! 💪",
            parse_mode="Markdown",
        )

        with open(pdf_path, "rb") as f:
            bot.send_document(chat_id, f, visible_file_name=pdf_fn, caption="📄 PDF-отчёт")
        with open(html_path, "rb") as f:
            bot.send_document(chat_id, f, visible_file_name=html_fn, caption="🌐 Интерактивный HTML")
        with open(txt_path, "rb") as f:
            bot.send_document(chat_id, f, visible_file_name=txt_fn, caption="📝 Транскрипция")

        for p in [pdf_path, html_path, txt_path]:
            if os.path.exists(p):
                os.remove(p)

    except Exception as ex:
        log.error(f"Error: {ex}", exc_info=True)
        bot.send_message(chat_id, f"😅 Ошибка: {str(ex)[:400]}\n\nПопробуй ещё раз!")
    finally:
        reset_session(chat_id)


# ═══════════════════════════════════════════════════
# HANDLERS
# ═══════════════════════════════════════════════════

@bot.message_handler(commands=["start", "help"])
def handle_start(m):
    bot.send_message(
        m.chat.id,
        "👋 Привет! Я *Цифровой Умник* – AI-аналитик встреч.\n\n"
        "Закидывай аудио/видео или ссылку YouTube/Google Drive.\n"
        "Когда всё загружено – жми /analyze\n\n"
        "🎙 *Принимаю:*\n"
        "• Аудио: mp3, wav, ogg, m4a, opus, flac\n"
        "• Видео: mp4, mov, avi, mkv, webm\n"
        "• Голосовые и видеосообщения\n"
        "• Ссылки YouTube / Google Drive\n\n"
        "🚀 Готов к работе!",
        parse_mode="Markdown",
    )


@bot.message_handler(commands=["analyze"])
def handle_analyze(m):
    s = get_session(m.chat.id)
    if s["processing"]:
        bot.send_message(m.chat.id, "⏳ Ещё обрабатываю. Подожди!")
        return
    if not s["files"] and not s["urls"]:
        bot.send_message(m.chat.id, "🤔 Файлов нет! Сначала скинь аудио/видео или ссылку.")
        return
    mk = types.InlineKeyboardMarkup(row_width=2)
    for code, (name, _) in LANGUAGES.items():
        mk.add(types.InlineKeyboardButton(name, callback_data=f"lang_{code}"))
    bot.send_message(m.chat.id, "🌍 На каком языке написать отчёт?", reply_markup=mk)


@bot.callback_query_handler(func=lambda c: c.data.startswith("lang_"))
def handle_language(c):
    s = get_session(c.message.chat.id)
    if s["processing"]:
        bot.answer_callback_query(c.id, "Уже обрабатываю!")
        return
    key = c.data[5:]
    name, lang_code = LANGUAGES.get(key, ("", "ru"))
    bot.answer_callback_query(c.id, f"Выбран: {name}")
    bot.edit_message_text(f"🌍 Язык отчёта: {name}", c.message.chat.id, c.message.message_id)
    threading.Thread(target=process_meeting, args=(c.message.chat.id, lang_code), daemon=True).start()


@bot.message_handler(content_types=["audio", "voice"])
def handle_audio(m):
    s = get_session(m.chat.id)
    if s["processing"]:
        return
    fid = m.audio.file_id if m.audio else m.voice.file_id
    fn = (m.audio.file_name if m.audio and m.audio.file_name else f"voice_{uuid.uuid4().hex[:6]}.ogg")
    s["files"].append({"id": fid, "name": fn})
    bot.send_message(m.chat.id, f"📎 Принято: *{fn}*\nЕщё? Или /analyze", parse_mode="Markdown")


@bot.message_handler(content_types=["video", "video_note"])
def handle_video(m):
    s = get_session(m.chat.id)
    if s["processing"]:
        return
    fid = m.video.file_id if m.video else m.video_note.file_id
    fn = (m.video.file_name if m.video and m.video.file_name else f"video_{uuid.uuid4().hex[:6]}.mp4")
    s["files"].append({"id": fid, "name": fn})
    bot.send_message(m.chat.id, f"📎 Принято: *{fn}*\nЕщё? Или /analyze", parse_mode="Markdown")


@bot.message_handler(content_types=["document"])
def handle_document(m):
    s = get_session(m.chat.id)
    if s["processing"]:
        return
    fn = m.document.file_name or "file"
    ext = os.path.splitext(fn)[1].lower()
    if ext in MEDIA_EXTS:
        s["files"].append({"id": m.document.file_id, "name": fn})
        bot.send_message(m.chat.id, f"📎 Принято: *{fn}*\nЕщё? Или /analyze", parse_mode="Markdown")
    else:
        bot.send_message(
            m.chat.id,
            f"🤔 *{fn}* – не аудио/видео.\nПоддерживаю: mp3, wav, m4a, mp4, mov...",
            parse_mode="Markdown",
        )


@bot.message_handler(content_types=["text"])
def handle_text(m):
    s = get_session(m.chat.id)
    if s["processing"]:
        return
    text = m.text.strip()

    # URL detection
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
            bot.send_message(m.chat.id, "🔗 Ссылка принята!\nЕщё? Или /analyze")
            return

    if re.match(r"https?://\S+", text):
        s["urls"].append(text)
        bot.send_message(m.chat.id, "🔗 Ссылка принята!\nЕщё? Или /analyze")
        return

    bot.send_message(
        m.chat.id,
        "👋 Скинь мне аудио/видео или ссылку YouTube/Google Drive.\n"
        "Когда всё загружено – жми /analyze 🚀",
    )


# ═══════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("🧠 Digital Smarty v2.0 starting...")
    log.info(f"Bot: @{bot.get_me().username}")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
