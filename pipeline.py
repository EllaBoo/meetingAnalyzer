"""
Digital Smarty v2.0 – Processing Pipeline
Transcription (Deepgram), Analysis (GPT-4o), Report Generation (PDF/HTML/TXT)
"""

import requests
import openai
import json
import os
import re
import uuid
import tempfile
import logging
from datetime import datetime

log = logging.getLogger("digital_smarty")
TMP = tempfile.gettempdir()


# ═══════════════════════════════════════════════════
# DOWNLOAD & AUDIO
# ═══════════════════════════════════════════════════

def download_from_youtube(url):
    import yt_dlp
    out = os.path.join(TMP, f"ds_yt_{uuid.uuid4().hex[:8]}.mp3")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out.replace(".mp3", ".%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
        "quiet": True, "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    base = out.replace(".mp3", "")
    for ext in [".mp3", ".m4a", ".wav", ".opus", ".webm"]:
        if os.path.exists(base + ext):
            return base + ext
    return out


def download_from_gdrive(url):
    fid = None
    for p in [r"/file/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)", r"/d/([a-zA-Z0-9_-]+)"]:
        m = re.search(p, url)
        if m:
            fid = m.group(1)
            break
    if not fid:
        raise ValueError("Cannot extract Google Drive file ID")
    dl_url = f"https://drive.google.com/uc?export=download&id={fid}&confirm=t"
    out = os.path.join(TMP, f"ds_gd_{uuid.uuid4().hex[:8]}.mp4")
    r = requests.get(dl_url, stream=True, timeout=300)
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return out


def download_from_url(url):
    if "youtube.com" in url or "youtu.be" in url:
        return download_from_youtube(url)
    elif "drive.google.com" in url:
        return download_from_gdrive(url)
    else:
        out = os.path.join(TMP, f"ds_dl_{uuid.uuid4().hex[:8]}.mp4")
        r = requests.get(url, timeout=180, stream=True)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return out


def extract_audio(file_path):
    from pydub import AudioSegment
    audio_exts = {".mp3", ".wav", ".ogg", ".m4a", ".opus", ".flac", ".aac", ".wma"}
    if os.path.splitext(file_path)[1].lower() in audio_exts:
        return file_path
    out = os.path.join(TMP, f"ds_audio_{uuid.uuid4().hex[:8]}.mp3")
    AudioSegment.from_file(file_path).export(out, format="mp3", bitrate="128k")
    return out


def split_audio(file_path, max_mb=90):
    from pydub import AudioSegment
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb <= max_mb:
        return [file_path]
    audio = AudioSegment.from_file(file_path)
    n = int(size_mb / max_mb) + 1
    chunk_dur = len(audio) // n
    chunks = []
    for i in range(n):
        cp = os.path.join(TMP, f"ds_ch_{uuid.uuid4().hex[:8]}_{i}.mp3")
        audio[i * chunk_dur : min((i + 1) * chunk_dur, len(audio))].export(cp, format="mp3", bitrate="128k")
        chunks.append(cp)
    log.info(f"Split audio into {n} chunks")
    return chunks


# ═══════════════════════════════════════════════════
# TRANSCRIPTION (Deepgram Nova-2)
# ═══════════════════════════════════════════════════

def format_ts(seconds):
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def transcribe_deepgram(audio_path, api_key):
    with open(audio_path, "rb") as f:
        data = f.read()
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/mpeg"}
    params = {
        "model": "nova-2", "smart_format": "true", "diarize": "true",
        "punctuate": "true", "paragraphs": "true", "utterances": "true",
        "language": "ru", "detect_language": "true",
    }
    log.info(f"Deepgram: sending {len(data)} bytes")
    r = requests.post("https://api.deepgram.com/v1/listen", headers=headers, params=params, data=data, timeout=600)
    if r.status_code != 200:
        raise Exception(f"Deepgram {r.status_code}: {r.text[:500]}")
    res = r.json()

    utterances = []
    if "results" in res and "utterances" in res["results"]:
        for u in res["results"]["utterances"]:
            utterances.append({
                "speaker": u.get("speaker", 0), "text": u.get("transcript", ""),
                "start": u.get("start", 0), "end": u.get("end", 0),
            })

    full_text, detected_lang = "", "unknown"
    if "results" in res and "channels" in res["results"]:
        ch = res["results"]["channels"]
        if ch and "alternatives" in ch[0]:
            full_text = ch[0]["alternatives"][0].get("transcript", "")
        if ch and "detected_language" in ch[0]:
            detected_lang = ch[0]["detected_language"]

    speaker_text = ""
    cur = None
    for u in utterances:
        lbl = f"Собеседник {u['speaker'] + 1}"
        ts = format_ts(u["start"])
        if u["speaker"] != cur:
            speaker_text += f"\n\n[{ts}] **{lbl}:**\n"
            cur = u["speaker"]
        speaker_text += u["text"] + " "

    speakers_count = len(set(u["speaker"] for u in utterances)) if utterances else 1
    duration = max((u["end"] for u in utterances), default=0)
    log.info(f"Transcribed: {speakers_count} speakers, {duration:.0f}s, lang={detected_lang}")

    return {
        "full_text": full_text, "speaker_transcript": speaker_text.strip(),
        "speakers_count": speakers_count, "detected_language": detected_lang,
        "duration_seconds": duration,
    }


def transcribe_file(file_path, deepgram_key):
    audio = extract_audio(file_path)
    if file_path != audio and os.path.exists(file_path):
        os.remove(file_path)
    chunks = split_audio(audio)
    results = []
    for ch in chunks:
        results.append(transcribe_deepgram(ch, deepgram_key))
        if ch != audio and os.path.exists(ch):
            os.remove(ch)
    if os.path.exists(audio):
        os.remove(audio)
    if len(results) == 1:
        return results[0]
    return {
        "full_text": "\n\n".join(t["full_text"] for t in results),
        "speaker_transcript": "\n\n--- (продолжение) ---\n\n".join(t["speaker_transcript"] for t in results),
        "speakers_count": max(t["speakers_count"] for t in results),
        "detected_language": results[0]["detected_language"],
        "duration_seconds": sum(t["duration_seconds"] for t in results),
    }


# ═══════════════════════════════════════════════════
# GPT-4o ANALYSIS
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """Ты – Цифровой Умник, AI-аналитик встреч. Ты слушаешь записи и превращаешь их в структурированные отчёты.
Тон: тёплый, с лёгким юмором. Как умный коллега, который серьёзно относится к работе, но не к себе.

ПРИНЦИПЫ:
1. Только факты из аудио. Не выдумывай.
2. Интерпретации маркируй «возможно», «судя по контексту».
3. Рекомендации = отдельная глава.
4. Адаптируйся к области обсуждения.
5. Используй цитаты.
6. «Собеседник 1, 2, 3...» если имена не прозвучали.
7. Оценивай идеи, не людей.

Ответ СТРОГО в JSON:
{"meeting_topic_short":"3-5 слов","passport":{"date":"...","duration_estimate":"...","participants_count":0,"participants":["Собеседник 1"],"format":"...","domain":"...","tone":"...","summary":"..."},"topics":[{"title":"...","description":"...","raised_by":"...","key_points":["..."],"positions":{"Собеседник 1":"..."},"outcome":"...","unresolved":["..."],"quotes":["..."]}],"decisions":[{"decision":"...","responsible":"...","deadline":"...","status":"accepted|pending|question"}],"unresolved_questions":[{"question":"...","reason":"...","assigned_to":"..."}],"dynamics":{"participation_balance":{"Собеседник 1":"45%"},"interaction_patterns":{"interruptions":"...","question_askers":["..."],"topic_initiators":["..."],"challengers":["..."]},"emotional_map":{"enthusiasm_moments":["..."],"tension_moments":["..."],"uncertainty_moments":["..."],"turning_points":["..."]},"unspoken":["..."]},"expert_recommendations":{"strengths":["..."],"attention_points":["..."],"recommendations":[{"what":"...","why":"...","how":"...","priority":"high|medium|low"}],"next_meeting_questions":["..."]}}"""


def analyze_meeting(transcript_data, language_code, openai_key):
    lang_map = {
        "ru": "Пиши отчёт на РУССКОМ.", "en": "Write report in ENGLISH.",
        "kk": "Есепті ҚАЗАҚ тілінде жаз.", "es": "Escribe en ESPAÑOL.",
        "zh": "用中文撰写报告。", "original": "Пиши на языке беседы.",
    }
    lang_note = lang_map.get(language_code, lang_map["original"])
    msg = (
        f"{lang_note}\n\nУчастников: {transcript_data['speakers_count']}\n"
        f"Язык: {transcript_data['detected_language']}\n"
        f"Длительность: {format_ts(transcript_data['duration_seconds'])}\n\n"
        f"ТРАНСКРИПЦИЯ:\n\n{transcript_data['speaker_transcript']}"
    )
    client = openai.OpenAI(api_key=openai_key)
    log.info("Sending to GPT-4o...")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}],
        temperature=0.4, max_tokens=16000,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ═══════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════

def esc(text):
    if not isinstance(text, str):
        text = str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_slug(analysis):
    raw = analysis.get("meeting_topic_short", "meeting")
    return re.sub(r"[^\w\s-]", "", raw).strip().replace(" ", "_")[:50]


def generate_pdf(analysis):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    try:
        pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DVB", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
        fn, fb = "DV", "DVB"
    except Exception:
        fn, fb = "Helvetica", "Helvetica-Bold"

    slug = make_slug(analysis)
    ds = datetime.now().strftime("%Y-%m-%d")
    fname = f"{slug}_{ds}_report.pdf"
    fpath = os.path.join(TMP, fname)
    doc = SimpleDocTemplate(fpath, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    S = getSampleStyleSheet()
    e = esc

    title_s = ParagraphStyle("T", parent=S["Title"], fontName=fb, fontSize=18, textColor=HexColor("#1a1a2e"), spaceAfter=5*mm)
    h1 = ParagraphStyle("H1", parent=S["Heading1"], fontName=fb, fontSize=13, textColor=HexColor("#16213e"), spaceBefore=6*mm, spaceAfter=3*mm)
    h2 = ParagraphStyle("H2", parent=S["Heading2"], fontName=fb, fontSize=11, textColor=HexColor("#0f3460"), spaceBefore=4*mm, spaceAfter=2*mm)
    bs = ParagraphStyle("B", parent=S["Normal"], fontName=fn, fontSize=9, leading=13, alignment=TA_JUSTIFY, spaceAfter=2*mm)
    lb = ParagraphStyle("L", parent=bs, fontName=fb, textColor=HexColor("#e94560"))
    ft = ParagraphStyle("F", parent=bs, fontSize=7, textColor=HexColor("#999"), alignment=TA_CENTER)

    st = []
    p = analysis.get("passport", {})
    st.append(Spacer(1, 2*cm))
    st.append(Paragraph("Цифровой Умник", title_s))
    st.append(Paragraph(f"<b>{e(p.get('summary',''))}</b>", bs))
    st.append(Spacer(1, 5*mm))

    rows = [[e(k), e(str(v))] for k, v in [
        ("Дата", p.get("date", "")), ("Длительность", p.get("duration_estimate", "")),
        ("Участники", p.get("participants_count", "")), ("Формат", p.get("format", "")),
        ("Область", p.get("domain", "")), ("Тон", p.get("tone", "")),
    ]]
    t = Table(rows, colWidths=[3.5*cm, 12.5*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), fb), ("FONTNAME", (1, 0), (1, -1), fn),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#e94560")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3*mm),
    ]))
    st.append(t)
    st.append(PageBreak())

    st.append(Paragraph("Темы", h1))
    for i, tp in enumerate(analysis.get("topics", []), 1):
        st.append(Paragraph(f"{i}. {e(tp.get('title', ''))}", h2))
        st.append(Paragraph(e(tp.get("description", "")), bs))
        for kp in tp.get("key_points", []):
            st.append(Paragraph(f"  – {e(kp)}", bs))
        for sp, pos in tp.get("positions", {}).items():
            st.append(Paragraph(f"<b>{e(sp)}:</b> {e(pos)}", bs))
        if tp.get("outcome"):
            st.append(Paragraph(f"<b>Итог:</b> {e(tp['outcome'])}", bs))
        for q in tp.get("quotes", [])[:3]:
            st.append(Paragraph(f"<i>«{e(q)}»</i>", bs))
        st.append(Spacer(1, 3*mm))

    decs = analysis.get("decisions", [])
    if decs:
        st.append(Paragraph("Решения", h1))
        for d in decs:
            ic = {"accepted": "[OK]", "pending": "[..]", "question": "[??]"}.get(d.get("status", ""), "[-]")
            st.append(Paragraph(f"{ic} <b>{e(d.get('decision', ''))}</b>", bs))

    dy = analysis.get("dynamics", {})
    if dy:
        st.append(PageBreak())
        st.append(Paragraph("Динамика", h1))
        for sp, pc in dy.get("participation_balance", {}).items():
            st.append(Paragraph(f"{e(sp)}: {e(pc)}", bs))
        em = dy.get("emotional_map", {})
        for k, label in [("enthusiasm_moments", "Энтузиазм"), ("tension_moments", "Напряжение"), ("turning_points", "Переломы")]:
            items = em.get(k, [])
            if items:
                st.append(Paragraph(f"<b>{label}:</b>", lb))
                for it in items:
                    st.append(Paragraph(f"  – {e(it)}", bs))

    rc = analysis.get("expert_recommendations", {})
    if rc:
        st.append(PageBreak())
        st.append(Paragraph("Рекомендации", h1))
        for s2 in rc.get("strengths", []):
            st.append(Paragraph(f"+ {e(s2)}", bs))
        for r in rc.get("recommendations", []):
            st.append(Paragraph(f"<b>{e(r.get('what', ''))}</b>", bs))
            if r.get("why"):
                st.append(Paragraph(f"  Почему: {e(r['why'])}", bs))
            if r.get("how"):
                st.append(Paragraph(f"  Как: {e(r['how'])}", bs))

    st.append(Spacer(1, 1*cm))
    st.append(Paragraph(f"Цифровой Умник • {ds}", ft))
    doc.build(st)
    log.info(f"PDF: {fname}")
    return fpath, fname


def generate_html(analysis):
    slug = make_slug(analysis)
    ds = datetime.now().strftime("%Y-%m-%d")
    fname = f"{slug}_{ds}_interactive.html"
    fpath = os.path.join(TMP, fname)
    p = analysis.get("passport", {})
    topics = analysis.get("topics", [])
    decs = analysis.get("decisions", [])
    dy = analysis.get("dynamics", {})
    rc = analysis.get("expert_recommendations", {})
    e = esc

    th = ""
    for i, t in enumerate(topics, 1):
        kps = "".join(f'<div class="kp">– {e(k)}</div>' for k in t.get("key_points", []))
        pos = "".join(f'<div><b>{e(s)}:</b> {e(v)}</div>' for s, v in t.get("positions", {}).items())
        quotes = "".join(f'<blockquote>«{e(q)}»</blockquote>' for q in t.get("quotes", [])[:3])
        th += f'<div class="tc"><div class="th" onclick="tog(this)"><span class="tn">{i}</span><span class="tt">{e(t.get("title",""))}</span><span class="ar">▼</span></div><div class="tb" style="display:none"><p>{e(t.get("description",""))}</p><p><b>Поднял(а):</b> {e(t.get("raised_by",""))}</p><p><b>Итог:</b> {e(t.get("outcome",""))}</p>{kps}{pos}{quotes}</div></div>'

    dh = ""
    for d in decs:
        ic = {"accepted": "✅", "pending": "⏳", "question": "❓"}.get(d.get("status", ""), "•")
        dh += f'<div class="di">{ic} <b>{e(d.get("decision",""))}</b><br><small>{e(d.get("responsible",""))} {e(d.get("deadline",""))}</small></div>'

    bh = ""
    for s, pc in dy.get("participation_balance", {}).items():
        n = int(re.sub(r"[^0-9]", "", str(pc)) or 0)
        bh += f'<div class="bb"><span class="bl">{e(s)}</span><div class="bc"><div class="bf" style="width:{n}%"></div></div><span>{e(pc)}</span></div>'

    rh = ""
    for r in rc.get("recommendations", []):
        ic = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(r.get("priority", ""), "•")
        rh += f'<div class="rc">{ic} <b>{e(r.get("what",""))}</b><br><small>{e(r.get("why",""))}</small></div>'

    html = f'''<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Цифровой Умник</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:system-ui,sans-serif;background:#f5f5f7;color:#1d1d1f;line-height:1.6;padding:16px}}
.w{{max-width:860px;margin:0 auto}}.hd{{background:linear-gradient(135deg,#1a1a2e,#0f3460);color:#fff;padding:30px;border-radius:14px;margin-bottom:20px}}.hd h1{{font-size:24px}}.hd p{{opacity:.85;margin-top:6px}}
.s{{background:#fff;border-radius:10px;padding:20px;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}.s h2{{font-size:18px;margin-bottom:12px;color:#16213e}}
.nt{{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}}.nb{{padding:6px 14px;border-radius:16px;background:#e8e8ed;cursor:pointer;font-size:13px;font-weight:500;border:none}}.nb:hover{{background:#d0d0d8}}.nb.a{{background:#e94560;color:#fff}}
.pn{{display:none}}.pn.a{{display:block}}
.pg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.pi .lb{{font-size:11px;color:#e94560;font-weight:600;text-transform:uppercase}}.pi .vl{{font-size:14px;margin-top:2px}}
.sb{{background:#f0f4ff;border-left:3px solid #e94560;padding:12px;border-radius:0 6px 6px 0;margin-top:12px}}
.tc{{border:1px solid #e8e8ed;border-radius:8px;margin-bottom:8px;overflow:hidden}}.th{{display:flex;align-items:center;padding:12px 16px;cursor:pointer;background:#fafafa;gap:10px}}.th:hover{{background:#f0f0f5}}
.tn{{background:#e94560;color:#fff;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;flex-shrink:0}}.tt{{flex:1;font-weight:600;font-size:14px}}.ar{{color:#999}}
.tb{{padding:16px;border-top:1px solid #e8e8ed}}.tb p{{margin-bottom:8px;font-size:13px}}.kp{{font-size:13px;padding:2px 0;color:#555}}
blockquote{{border-left:3px solid #e94560;padding:6px 14px;margin:6px 0;color:#666;font-style:italic;background:#fafafa;border-radius:0 6px 6px 0;font-size:13px}}
.di{{padding:10px;border-radius:6px;margin-bottom:6px;background:#f8f8fa;font-size:14px}}.di small{{color:#888}}
.bb{{display:flex;align-items:center;margin-bottom:8px}}.bl{{width:120px;font-size:13px}}.bc{{flex:1;height:20px;background:#e8e8ed;border-radius:10px;overflow:hidden;margin:0 10px}}.bf{{height:100%;background:linear-gradient(90deg,#e94560,#0f3460);border-radius:10px}}
.rc{{padding:12px;border-radius:8px;margin-bottom:8px;background:#fafafa;border:1px solid #e8e8ed;font-size:14px}}
.ft{{text-align:center;padding:16px;color:#999;font-size:12px}}
</style></head><body><div class="w">
<div class="hd"><h1>🧠 Цифровой Умник</h1><p>{e(p.get("summary",""))}</p></div>
<div class="nt"><button class="nb a" onclick="go('ov')">📋 Обзор</button><button class="nb" onclick="go('tp')">🎯 Темы ({len(topics)})</button><button class="nb" onclick="go('dc')">✅ Решения</button><button class="nb" onclick="go('dy')">📊 Динамика</button><button class="nb" onclick="go('rc')">💡 Рекомендации</button></div>
<div id="p-ov" class="pn a"><div class="s"><div class="pg">
<div class="pi"><div class="lb">Дата</div><div class="vl">{e(p.get("date",""))}</div></div>
<div class="pi"><div class="lb">Длительность</div><div class="vl">{e(p.get("duration_estimate",""))}</div></div>
<div class="pi"><div class="lb">Участники</div><div class="vl">{e(str(p.get("participants_count","")))}</div></div>
<div class="pi"><div class="lb">Формат</div><div class="vl">{e(p.get("format",""))}</div></div>
<div class="pi"><div class="lb">Область</div><div class="vl">{e(p.get("domain",""))}</div></div>
<div class="pi"><div class="lb">Тон</div><div class="vl">{e(p.get("tone",""))}</div></div>
</div><div class="sb">{e(p.get("summary",""))}</div></div></div>
<div id="p-tp" class="pn"><div class="s"><h2>🎯 Темы</h2>{th}</div></div>
<div id="p-dc" class="pn"><div class="s"><h2>✅ Решения</h2>{dh or "<p>Решения не зафиксированы</p>"}</div></div>
<div id="p-dy" class="pn"><div class="s"><h2>📊 Динамика</h2>{f"<h3>Баланс участия</h3>{bh}" if bh else ""}</div></div>
<div id="p-rc" class="pn"><div class="s"><h2>💡 Рекомендации</h2>{"".join(f"<div>✅ {e(s)}</div>" for s in rc.get("strengths",[]))}{rh}</div></div>
<div class="ft">Цифровой Умник • {ds}</div></div>
<script>function go(id){{document.querySelectorAll('.pn').forEach(x=>x.classList.remove('a'));document.querySelectorAll('.nb').forEach(x=>x.classList.remove('a'));document.getElementById('p-'+id).classList.add('a');event.target.classList.add('a')}}
function tog(el){{var b=el.nextElementSibling;b.style.display=b.style.display==='none'?'block':'none'}}</script></body></html>'''

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"HTML: {fname}")
    return fpath, fname


def generate_txt(analysis, transcript_text):
    slug = make_slug(analysis)
    ds = datetime.now().strftime("%Y-%m-%d")
    fname = f"{slug}_{ds}_transcription.txt"
    fpath = os.path.join(TMP, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(f"ТРАНСКРИПЦИЯ\n{'='*50}\nТема: {analysis.get('meeting_topic_short','')}\nДата: {ds}\n{'='*50}\n\n")
        f.write(transcript_text)
    return fpath, fname
