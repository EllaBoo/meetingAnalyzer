FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-freefont-ttf \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fonts AFTER COPY so they never get overwritten
RUN mkdir -p fonts && \
    cp /usr/share/fonts/truetype/freefont/FreeSans.ttf fonts/ && \
    cp /usr/share/fonts/truetype/freefont/FreeSansBold.ttf fonts/ && \
    cp /usr/share/fonts/truetype/freefont/FreeSansOblique.ttf fonts/ && \
    cp /usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf fonts/ && \
    curl -sL -o fonts/NotoSansSC-Regular.ttf \
    "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"

CMD ["python", "bot.py"]
