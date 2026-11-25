import os
import json
import time
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

GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.0-flash", "gemini-2.0-flash"]


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


def parse_json_safely(raw_text):
    if not raw_text:
        return {}
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except Exception:
                pass
    return {}


def post_gemini_with_fallback(gemini_key, payload, timeout_sec=60):
    last_err = "Sin respuesta de Gemini"
    for model_name in GEMINI_MODELS:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={gemini_key}"
        )
        for attempt in range(2):
            try:
                res = requests.post(endpoint, json=payload, timeout=timeout_sec)
            except requests.Timeout:
                last_err = f"Tiempo de espera agotado en {model_name}"
                time.sleep(1.5)
                continue
            except Exception as req_err:
                last_err = str(req_err)
                break

            if res.ok:
                return res.json()

            err_json = res.json() if res.content else {}
            last_err = err_json.get("error", {}).get("message", res.text or f"HTTP {res.status_code}")
            lower_err = last_err.lower()

            if res.status_code in (429, 503) or "high demand" in lower_err or "overloaded" in lower_err:
                time.sleep(2 * (attempt + 1))
                continue
            break

    raise RuntimeError(last_err)


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
                "coverUrl": (
                    sp.get("album", {}).get("images", [{}])[0].get("url")
                    or dz.get("album", {}).get("cover_medium", "")
                ),
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

        data = post_gemini_with_fallback(gemini_key, payload, timeout_sec=35)
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        raw = "\n".join(p.get("text", "") for p in parts)
        parsed = parse_json_safely(raw)

        title = str(parsed.get("title", "")).strip()
        if title:
            return {
                "recognized": True,
                "title": title,
                "artist": str(parsed.get("artist", "Artista Detectado")).strip() or "Artista Detectado",
                "album": str(parsed.get("genre", "IA Acústica")).strip() or "IA Acústica",
                "lyricsSnippet": str(parsed.get("lyricsHeard", "")).strip(),
                "source": "Gemini Audio"
            }
    except Exception as e:
        print("Gemini audio error:", e)
    return None


def analyze_youtube_direct_with_gemini(yt_url, gemini_key, start_min, end_min):
    """Respaldo directo: Google procesa la URL de YouTube sin descarga local."""
    prompt = f"""Escucha el audio de este video de YouTube únicamente entre el minuto {start_min} y el minuto {end_min}.
Identifica las canciones reales que suenan en orden cronológico dentro de ese rango.
NO inventes temas ni incluyas canciones fuera de ese rango.
Devuelve ÚNICAMENTE un objeto JSON con este formato exacto:
{{
  "videoTitle": "Título del video o mix",
  "tracks": [
    {{
      "timestampSec": {start_min * 60},
      "title": "Nombre real de la canción",
      "artist": "Artista real",
      "genre": "Género"
    }}
  ]
}}"""

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {
                    "fileData": {
                        "mimeType": "video/mp4",
                        "fileUri": yt_url
                    }
                },
                {"text": prompt}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    data = post_gemini_with_fallback(gemini_key, payload, timeout_sec=75)
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    raw = "\n".join(p.get("text", "") for p in parts)
    parsed = parse_json_safely(raw)

    min_allowed_sec = start_min * 60
    max_allowed_sec = end_min * 60

    tracks = []
    for t in parsed.get("tracks", []):
        title = str(t.get("title", "")).strip()
        if not title:
            continue
        sec = int(t.get("timestampSec", min_allowed_sec) or min_allowed_sec)
        if sec < min_allowed_sec or sec > max_allowed_sec:
            continue
        tracks.append({
            "timestampSec": sec,
            "title": title,
            "artist": str(t.get("artist", "Artista")).strip() or "Artista",
            "genre": str(t.get("genre", "Mix")).strip() or "Mix",
            "lyricsSnippet": "Reconocido directo desde YouTube",
            "coverUrl": "",
            "previewUrl": "",
            "spotifyUrl": "",
            "source": "Gemini YouTube Directo"
        })

    tracks.sort(key=lambda item: item["timestampSec"])
    return str(parsed.get("videoTitle", "Mix de YouTube")).strip(), tracks


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "SONICDJ Cloud Server v5"})


@app.route("/api/scan-youtube", methods=["POST"])
def scan_youtube():
    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    start_min = max(0, int(data.get("start_min", 0)))
    end_min = max(start_min + 1, int(data.get("end_min", 10)))
    interval_sec = max(15, int(data.get("interval_sec", 60)))
    sample_sec = max(5, min(15, int(data.get("sample_sec", 8))))

    gemini_key = (data.get("gemini_key") or os.environ.get("GEMINI_API_KEY") or "").strip()
    audd_key = (data.get("audd_key") or os.environ.get("AUDD_API_KEY") or "").strip()
    genre_hint = (data.get("genre_hint") or "").strip()

    if not url:
        return jsonify({"error": "Falta el enlace de YouTube."}), 400
    if not gemini_key and not audd_key:
        return jsonify({"error": "Falta configurar GEMINI_API_KEY en Render."}), 400

    tmp_dir = tempfile.mkdtemp(prefix="sonicdj_")
    try:
        out_template = os.path.join(tmp_dir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "noplaylist": True,
            "extractor_args": {"youtube": {"player_client": ["ios", "android"]}},
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

    except Exception as ydl_err:
        print("yt-dlp bloqueado en Render, usando Gemini Directo:", ydl_err)
        if gemini_key:
            try:
                video_title, tracks = analyze_youtube_direct_with_gemini(url, gemini_key, start_min, end_min)
                return jsonify({
                    "status": "ok",
                    "title": video_title,
                    "duration": end_min * 60,
                    "tracks": tracks
                })
            except Exception as gem_err:
                return jsonify({"error": f"YouTube bloqueó la descarga y el respaldo falló: {str(gem_err)}"}), 500
        return jsonify({"error": f"Error procesando YouTube: {str(ydl_err)}"}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
