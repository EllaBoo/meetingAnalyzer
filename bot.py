"""
Digital Smarty v3.0 – Telegram Bot (Pyrogram)
AI Meeting Analyzer with Deepgram + GPT-4o
Supports files up to 2GB via Telegram MTProto

v3.0 changes:
- Progress bar via single message edit (instead of 4 separate messages)
- Preview summary in chat before sending files
- Transcript caching for retranslation (no re-transcription)
- Timer showing processing duration
"""

import os
import re
import uuid
import asyncio
import logging
import tempfile
import time

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
        sessions[cid] = {
            "files": [], "urls": [], "processing": False,
            "last_analysis": None, "last_transcript": None,
            "last_transcript_data": None,  # v3: cached transcript for retranslation
        }
    return sessions[cid]


def reset_session(cid):
    old = sessions.get(cid, {})
    sessions[cid] = {
        "files": [], "urls": [], "processing": False,
        "last_analysis": old.get("last_analysis"),
        "last_transcript": old.get("last_transcript"),
        "last_transcript_data": old.get("last_transcript_data"),  # v3: preserve cache
    }


# ═══════════════════════════════════════════════════
# PROGRESS BAR (v3)
# ═══════════════════════════════════════════════════

STEP_ICONS = {
    "done": "✅",
    "active": "⏳",
    "pending": "⬜",
}


def build_progress_text(steps, current_step, n_files=1, extra_info=None):
    """Build progress message with step indicators.

    steps: list of (step_key, label)
    current_step: index of active step (0-based)
    extra_info: optional dict with extra info per step (e.g. speakers count)
    """
    lines = [f"🧠 **Обработка** ({n_files} источник(ов))\n"]
    for i, (key, label) in enumerate(steps):
        if i < current_step:
            icon = STEP_ICONS["done"]
            suffix = ""
            if extra_info and key in extra_info:
                suffix = f"  _{extra_info[key]}_"
            lines.append(f"{icon} {label}{suffix}")
        elif i == current_step:
            lines.append(f"{STEP_ICONS['active']} {label}...")
        else:
            lines.append(f"{STEP_ICONS['pending']} {label}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════
# PREVIEW (v3)
# ═══════════════════════════════════════════════════

def build_preview(analysis):
    """Build a short text preview of the analysis for chat."""
    topic = analysis.get("meeting_topic_short", "")
    passport = analysis.get("passport", {})

    participants = passport.get("participants_count", "?")
    duration = passport.get("duration_estimate", "?")
    tone = passport.get("tone", "")
    domain = passport.get("domain", "")

    n_topics = len(analysis.get("topics", []))
    n_decisions = len(analysis.get("decisions", []))
    n_actions = len(analysis.get("action_items", []))

    # Executive summary or passport summary
    summary = analysis.get("executive_summary", "") or passport.get("summary", "")
    if len(summary) > 200:
        summary = summary[:197] + "..."

    # Key decision (first one)
    key_decision = ""
    decisions = analysis.get("decisions", [])
    if decisions:
        d = decisions[0].get("decision", "")
        if d:
            key_decision = f"\n🎯 Ключевое решение: _{d}_"

    # Main insight from conclusion
    insight = ""
    conclusion = analysis.get("conclusion", {})
    if conclusion and conclusion.get("main_insight"):
        insight = f"\n💡 _{conclusion['main_insight']}_"

    lines = [
        f"📋 **{topic}**",
        f"👥 {participants} уч. | ⏱ {duration} | 🎭 {tone}",
    ]
    if domain:
        lines.append(f"🏷 {domain}")
    lines.append(f"🎯 {n_topics} тем | ✅ {n_decisions} решений | 📌 {n_actions} задач")
    if summary:
        lines.append(f"\n{summary}")
    if key_decision:
        lines.append(key_decision)
    if insight:
        lines.append(insight)

    return "\n".join(lines)


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
# PROCESSING (v3: progress bar + preview + timer)
# ═══════════════════════════════════════════════════

async def process_meeting(client, chat_id, lang_code):
    s = get_session(chat_id)
    start_time = time.time()
    try:
        s["processing"] = True
        n_files = len(s["files"]) + len(s["urls"])

        STEPS = [
            ("download", "Скачиваю и извлекаю аудио"),
            ("transcribe", "Транскрибирую (Deepgram Nova-2)"),
            ("analyze", "Анализирую как эксперт (GPT-4o)"),
            ("generate", "Генерирую отчёты"),
        ]

        # Send initial progress message
        progress_msg = await client.send_message(
            chat_id,
            build_progress_text(STEPS, 0, n_files) + "\n\n☕ Сходи за кофе, я тут пока послушаю...",
        )

        all_transcripts = []
        extra_info = {}

        # Step 0: Download
        for fi in s["files"]:
            path = await download_tg_file(fi["msg"], fi["name"])
            # Step 1: Transcribe
            await progress_msg.edit_text(
                build_progress_text(STEPS, 1, n_files, extra_info)
                + "\n\n☕ Сходи за кофе, я тут пока послушаю...",
            )
            t = await asyncio.to_thread(transcribe_file, path, DEEPGRAM_API_KEY)
            all_transcripts.append(t)

        for url in s["urls"]:
            path = await asyncio.to_thread(download_from_url, url)
            await progress_msg.edit_text(
                build_progress_text(STEPS, 1, n_files, extra_info)
                + "\n\n☕ Сходи за кофе, я тут пока послушаю...",
            )
            t = await asyncio.to_thread(transcribe_file, path, DEEPGRAM_API_KEY)
            all_transcripts.append(t)

        if not all_transcripts:
            await progress_msg.edit_text("😅 Не удалось обработать файлы. Попробуй ещё раз!")
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

        extra_info["transcribe"] = f"{merged['speakers_count']} спикеров, {format_ts(merged['duration_seconds'])}"

        # v3: Cache transcript data for retranslation (no re-transcription!)
        s["last_transcript_data"] = merged

        # Step 2: Analyze
        await progress_msg.edit_text(
            build_progress_text(STEPS, 2, n_files, extra_info)
            + "\n\n🧠 Это самая умная часть...",
        )
        analysis = await asyncio.to_thread(analyze_meeting, merged, lang_code, OPENAI_API_KEY)

        # Step 3: Generate
        await progress_msg.edit_text(
            build_progress_text(STEPS, 3, n_files, extra_info),
        )
        pdf_path, pdf_fn = await asyncio.to_thread(generate_pdf, analysis, lang_code)
        html_path, html_fn = await asyncio.to_thread(generate_html, analysis, merged["speaker_transcript"], lang_code)
        txt_path, txt_fn = await asyncio.to_thread(generate_txt, analysis, merged["speaker_transcript"])

        # Save for re-translation
        s["last_analysis"] = analysis
        s["last_transcript"] = merged["speaker_transcript"]

        # Final progress: all done + timer
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_str = f"{mins} мин {secs} сек" if mins > 0 else f"{secs} сек"

        await progress_msg.edit_text(
            build_progress_text(STEPS, len(STEPS), n_files, extra_info)
            + f"\n\n✅ **Готово за {time_str}**",
        )

        # v3: Preview before files
        preview = build_preview(analysis)
        await client.send_message(chat_id, preview)

        # Send files
        await client.send_message(
            chat_id,
            "📄 PDF – структурированный отчёт (для начальства)\n"
            "🌐 HTML – интерактивный разбор (для души)\n"
            "📝 TXT – полная транскрипция (для параноиков)",
        )

        await client.send_document(chat_id, pdf_path, file_name=pdf_fn, caption="📄 PDF-отчёт")
        await client.send_document(chat_id, html_path, file_name=html_fn, caption="🌐 Интерактивный HTML")
        await client.send_document(chat_id, txt_path, file_name=txt_fn, caption="📝 Транскрипция")

        # Offer translation
        translate_buttons = []
        for code, (name, _) in LANGUAGES.items():
            if code != lang_code:
                translate_buttons.append([InlineKeyboardButton(name, callback_data=f"retranslate_{code}")])
        await client.send_message(
            chat_id,
            "🌍 **Хочешь этот же отчёт на другом языке?**\nВыбери язык или скинь новую запись:",
            reply_markup=InlineKeyboardMarkup(translate_buttons),
        )

        for p in [pdf_path, html_path, txt_path]:
            if os.path.exists(p):
                os.remove(p)

    except Exception as ex:
        log.error(f"Error: {ex}", exc_info=True)
        await client.send_message(chat_id, f"😅 Упс, что-то пошло не так: {str(ex)[:400]}\n\nНо я не сдаюсь – попробуй ещё раз!")
    finally:
        reset_session(chat_id)


# ═══════════════════════════════════════════════════
# HANDLERS
# ═══════════════════════════════════════════════════

@app.on_message(filters.command(["start", "help"]))
async def handle_start(client, message: Message):
    await message.reply(
        "👋 Привет! Я **Цифровой Умник** – твой AI-аналитик встреч.\n\n"
        "Закидывай мне записи своих встреч, брейнштормов и созвонов, "
        "а я превращу этот хаос в структурированный отчёт с экспертным анализом. "
        "Да, я тот самый коллега, который реально слушает на совещаниях ☕\n\n"
        "🎙 **Принимаю:**\n"
        "• Аудио: mp3, wav, ogg, m4a, opus, flac\n"
        "• Видео: mp4, mov, avi, mkv, webm\n"
        "• Голосовые и видеосообщения\n"
        "• Ссылки YouTube / Google Drive\n"
        "• Файлы до 2 ГБ 💪\n\n"
        "📤 **На выходе:**\n"
        "• PDF – красивый отчёт\n"
        "• HTML – интерактивный разбор\n"
        "• TXT – полная транскрипция\n\n"
        "Скинь файл и жми /analyze – остальное я беру на себя 🚀",
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


@app.on_callback_query(filters.regex(r"^retranslate_"))
async def handle_retranslate(client, callback: CallbackQuery):
    chat_id = callback.message.chat.id
    s = get_session(chat_id)
    lang_code_key = callback.data.replace("retranslate_", "")

    # v3: Use cached transcript_data (no re-transcription!)
    cached_td = s.get("last_transcript_data")
    if not cached_td and not s.get("last_transcript"):
        await callback.answer("Нет данных для перевода. Скинь новую запись!")
        return

    if s["processing"]:
        await callback.answer("Уже обрабатываю!")
        return

    s["processing"] = True
    lang_name, lang_code = LANGUAGES.get(lang_code_key, ("", "ru"))
    start_time = time.time()

    await callback.message.edit_text(
        f"🌍 Язык: **{lang_name}**\n\n"
        f"✅ Транскрипция (из кеша)\n"
        f"⏳ Анализирую на новом языке...",
    )
    await callback.answer()

    try:
        # v3: Reuse cached transcript data instead of re-building from analysis
        if cached_td:
            transcript_data = cached_td
        else:
            # Fallback for old sessions without cache
            transcript_data = {
                "speakers_count": s["last_analysis"].get("passport", {}).get("participants_count", 2),
                "detected_language": s["last_analysis"].get("passport", {}).get("tone", ""),
                "duration_seconds": 0,
                "speaker_transcript": s["last_transcript"],
            }

        analysis = await asyncio.to_thread(analyze_meeting, transcript_data, lang_code, OPENAI_API_KEY)

        await callback.message.edit_text(
            f"🌍 Язык: **{lang_name}**\n\n"
            f"✅ Транскрипция (из кеша)\n"
            f"✅ Анализ\n"
            f"⏳ Генерирую отчёты...",
        )

        pdf_path, pdf_fn = await asyncio.to_thread(generate_pdf, analysis, lang_code)
        html_path, html_fn = await asyncio.to_thread(generate_html, analysis, s["last_transcript"], lang_code)

        # Save new analysis
        s["last_analysis"] = analysis

        elapsed = time.time() - start_time
        secs = int(elapsed)

        await callback.message.edit_text(
            f"🌍 Язык: **{lang_name}**\n\n"
            f"✅ Транскрипция (из кеша)\n"
            f"✅ Анализ\n"
            f"✅ Отчёты\n\n"
            f"✅ **Готово за {secs} сек**",
        )

        # v3: Preview for retranslation too
        preview = build_preview(analysis)
        await client.send_message(chat_id, preview)

        await client.send_document(chat_id, pdf_path, file_name=pdf_fn, caption=f"📄 PDF ({lang_name})")
        await client.send_document(chat_id, html_path, file_name=html_fn, caption=f"🌐 HTML ({lang_name})")

        # Offer more languages
        translate_buttons = []
        for code, (name, _) in LANGUAGES.items():
            if code != lang_code_key:
                translate_buttons.append([InlineKeyboardButton(name, callback_data=f"retranslate_{code}")])
        await client.send_message(
            chat_id,
            "✅ Готово! Ещё язык или скинь новую запись 💪",
            reply_markup=InlineKeyboardMarkup(translate_buttons),
        )

        for p in [pdf_path, html_path]:
            if os.path.exists(p):
                os.remove(p)

    except Exception as ex:
        log.error(f"Retranslate error: {ex}", exc_info=True)
        await client.send_message(chat_id, f"😅 Ошибка перевода: {str(ex)[:300]}")
    finally:
        s["processing"] = False


@app.on_callback_query(filters.regex(r"^start_analyze$"))
async def handle_start_analyze(client, callback: CallbackQuery):
    s = get_session(callback.message.chat.id)
    if s["processing"]:
        await callback.answer("Уже обрабатываю!")
        return
    if not s["files"] and not s["urls"]:
        await callback.answer("Сначала скинь файл!")
        return
    buttons = []
    for code, (name, _) in LANGUAGES.items():
        buttons.append([InlineKeyboardButton(name, callback_data=f"lang_{code}")])
    await callback.message.edit_text(
        "🌍 На каком языке написать отчёт?",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
    await callback.answer()


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
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Анализировать", callback_data="start_analyze")]])
    await message.reply(f"📎 Принято: **{fn}**\nЕщё файлы? Или жми кнопку:", reply_markup=kb)


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
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Анализировать", callback_data="start_analyze")]])
    await message.reply(f"📎 Принято: **{fn}**\nЕщё файлы? Или жми кнопку:", reply_markup=kb)


@app.on_message(filters.document)
async def handle_document(client, message: Message):
    s = get_session(message.chat.id)
    if s["processing"]:
        return
    fn = message.document.file_name or "file"
    ext = os.path.splitext(fn)[1].lower()
    if ext in MEDIA_EXTS:
        s["files"].append({"msg": message, "name": fn})
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Анализировать", callback_data="start_analyze")]])
        await message.reply(f"📎 Принято: **{fn}**\nЕщё файлы? Или жми кнопку:", reply_markup=kb)
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
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Анализировать", callback_data="start_analyze")]])
            await message.reply("🔗 Ссылка принята!\nЕщё? Или жми кнопку:", reply_markup=kb)
            return

    if re.match(r"https?://\S+", text):
        s["urls"].append(text)
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Анализировать", callback_data="start_analyze")]])
        await message.reply("🔗 Ссылка принята!\nЕщё? Или жми кнопку:", reply_markup=kb)
        return

    await message.reply(
        "👋 Скинь мне аудио/видео или ссылку YouTube/Google Drive.\n"
        "Когда всё загружено – жми /analyze и я займусь делом 🚀",
    )


# ═══════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("🧠 Digital Smarty v3.0 starting (Pyrogram MTProto)...")
    log.info(f"API_ID={API_ID}, TMP={TMP}")
    log.info("Flushed old updates. Starting polling...")
    app.run()
