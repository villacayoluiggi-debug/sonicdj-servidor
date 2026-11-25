import os
import json
import base64
import shutil
import tempfile
import subprocess
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp

app = Flask(__name__)
CORS(app)


def get_audio_duration(filepath):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(float(res.stdout.strip()))
    except Exception:
        return 0


def extract_wav_clip(input_path, start_sec, duration_sec, out_wav):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        out_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000


def recognize_audd(wav_path, audd_key):
    if not audd_key:
        return None
    try:
        with open(wav_path, "rb") as f:
            res = requests.post(
                "https://api.audd.io/",
                data={"api_token": audd_key, "return": "spotify,deezer"},
                files={"file": ("sample.wav", f, "audio/wav")},
                timeout=20
            )
        data = res.json()
        if data.get("status") == "success" and data.get("result"):
            r = data["result"]
            sp = r.get("spotify") or {}
            dz = r.get("deezer") or {}
            return {
                "recognized": True,
                "title": r.get("title", ""),
                "artist": r.get("artist", ""),
                "album": r.get("album", "AudD Shazam"),
                "coverUrl": (sp.get("album", {}).get("images", [{}])[0].get("url")
                             or dz.get("album", {}).get("cover_medium", "")),
                "previewUrl": sp.get("preview_url") or dz.get("preview", ""),
                "spotifyUrl": sp.get("external_urls", {}).get("spotify", ""),
                "source": "Shazam / AudD"
            }
    except Exception as e:
        print("AudD error:", e)
    return None


def recognize_gemini(wav_path, gemini_key, timestamp_sec, genre_hint=""):
    if not gemini_key:
        return None
    try:
        with open(wav_path, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode("utf-8")

        prompt = f"""Escucha este fragmento de audio real (minuto {timestamp_sec}s).
NO inventes canciones. Transcribe la letra que escuches e identifica el título y artista reales.
{f'Pista de género: {genre_hint}' if genre_hint else ''}
Devuelve SOLO JSON:
{{"recognized": true, "title": "Título real o vacío", "artist": "Artista real o vacío", "lyricsHeard": "Letra escuchada", "genre": "Género"}}"""

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "audio/wav", "data": b64_audio}}
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        res = requests.post(url, json=payload, timeout=30)
        if res.ok:
            raw = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw.replace("```json", "").replace("```", "").strip())
            if parsed.get("title"):
                return {
                    "recognized": True,
                    "title": parsed.get("title", "").strip(),
                    "artist": parsed.get("artist", "Artista Detectado").strip(),
                    "album": parsed.get("genre", "IA Acústica"),
                    "lyricsSnippet": parsed.get("lyricsHeard", ""),
                    "source": "Gemini Audio"
                }
    except Exception as e:
        print("Gemini error:", e)
    return None


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "SONICDJ Cloud Server"})


@app.route("/api/scan-youtube", methods=["POST"])
def scan_youtube():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    start_min = max(0, int(data.get("start_min", 0)))
    end_min = max(start_min + 1, int(data.get("end_min", 10)))
    interval_sec = max(15, int(data.get("interval_sec", 60)))
    sample_sec = max(5, min(15, int(data.get("sample_sec", 8))))

    # Usa la key del usuario o la key guardada en Render automáticamente
    gemini_key = (data.get("gemini_key") or os.environ.get("GEMINI_API_KEY") or "").strip()
    audd_key = (data.get("audd_key") or os.environ.get("AUDD_API_KEY") or "").strip()
    genre_hint = (data.get("genre_hint") or "").strip()

    if not url:
        return jsonify({"error": "Falta el enlace de YouTube."}), 400
    if not gemini_key and not audd_key:
        return jsonify({"error": "No hay API Key configurada en el servidor ni en la app."}), 400

    tmp_dir = tempfile.mkdtemp(prefix="sonicdj_")
    try:
        out_template = os.path.join(tmp_dir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get("title", "Mix de YouTube")
            downloaded_file = ydl.prepare_filename(info)

        total_duration = get_audio_duration(downloaded_file) or int(info.get("duration") or 0)
        start_sec = start_min * 60
        end_sec = min(end_min * 60, total_duration) if total_duration > 0 else end_min * 60

        tracks = []
        for t in range(start_sec, end_sec + 1, interval_sec):
            if total_duration > 0 and t >= total_duration - 3:
                break
            wav_clip = os.path.join(tmp_dir, f"clip_{t}.wav")
            if not extract_wav_clip(downloaded_file, t, sample_sec, wav_clip):
                continue

            match = recognize_audd(wav_clip, audd_key)
            if not match:
                match = recognize_gemini(wav_clip, gemini_key, t, genre_hint)

            if match and match.get("recognized"):
                tracks.append({
                    "timestampSec": t,
                    "title": match["title"],
                    "artist": match["artist"],
                    "genre": match.get("album", "Detectado"),
                    "lyricsSnippet": match.get("lyricsSnippet", f"Detectado en {t}s"),
                    "coverUrl": match.get("coverUrl", ""),
                    "previewUrl": match.get("previewUrl", ""),
                    "spotifyUrl": match.get("spotifyUrl", ""),
                    "source": match.get("source", "Servidor Nube")
                })

        return jsonify({
            "status": "ok",
            "title": video_title,
            "duration": total_duration,
            "tracks": tracks
        })

    except Exception as e:
        return jsonify({"error": f"Error procesando YouTube: {str(e)}"}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)