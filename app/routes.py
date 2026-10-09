from flask import Blueprint, jsonify, render_template, request, send_from_directory, send_file
from .providers import PROVIDERS, run_model
from .evaluator import evaluate, summarize
from .reporting import save_json, save_html, save_pdf

import io
import os
import tempfile


bp = Blueprint("main", __name__)


# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------

@bp.get("/")
def index():
    return render_template(
        "index.html",
        providers=PROVIDERS
    )


# ---------------------------------------------------------
# VOICE TESTER PAGE
# ---------------------------------------------------------

@bp.get("/voice")
def voice():
    return render_template(
        "voice.html",
        providers=PROVIDERS
    )


# ---------------------------------------------------------
# PROVIDERS API
# ---------------------------------------------------------

@bp.get("/api/providers")
def providers():
    return jsonify(PROVIDERS)


# ---------------------------------------------------------
# RUN MODEL + EVALUATION
# ---------------------------------------------------------

@bp.post("/api/run")
def run():

    data = request.get_json(force=True) or {}

    provider = data.get("provider", "ollama")

    model = (
        data.get("model")
        or PROVIDERS.get(provider, {}).get("default_model")
    )

    prompt = (data.get("prompt") or "").strip()

    api_key = data.get("api_key") or None

    reference = data.get("reference") or None

    if provider not in PROVIDERS:
        return jsonify({"error": "Unsupported provider"}), 400

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:

        text, latency, usage = run_model(
            provider,
            model,
            prompt,
            api_key
        )

        evaluation = evaluate(prompt, text, reference)

        result = {
            "provider": provider,
            "provider_label": PROVIDERS[provider]["label"],
            "model": model,
            "prompt": prompt,
            "response": text,
            "latency_ms": latency,
            "usage": usage,
            "evaluation": evaluation
        }

        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


# ---------------------------------------------------------
# VOICE  —  TRANSCRIBE AUDIO  (STT)
# ---------------------------------------------------------

@bp.post("/api/voice/transcribe")
def voice_transcribe():
    """
    Accepts a multipart/form-data request with:
      - audio   : the recorded audio file (webm / wav / mp4 …)
      - provider: 'groq' | 'openai'  (default: groq)
      - api_key : optional override key
    Returns: { "transcript": "<text>" }
    """

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    stt_provider = request.form.get("stt_provider", "groq")
    api_key = request.form.get("api_key") or None

    # Save to a temp file so SDKs can read it
    suffix = ".webm"
    original = audio_file.filename or ""
    if "." in original:
        suffix = "." + original.rsplit(".", 1)[-1]

        tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        transcript = _transcribe(tmp_path, stt_provider, api_key)

    except Exception as e:
        return jsonify({"error": str(e)}), 502   # ← now returns proper JSON

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return jsonify({"transcript": transcript})


def _transcribe(path, provider, api_key):
    """Call the appropriate Whisper endpoint and return the transcript."""

    # ---- Groq Whisper ----
    if provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("Groq API key is missing.")
        from groq import Groq
        client = Groq(api_key=key)
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(path), f),
                model="whisper-large-v3",
            )
        return result.text

    # ---- OpenAI Whisper ----
    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OpenAI API key is missing.")
        from openai import OpenAI
        client = OpenAI(api_key=key)
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
            )
        return result.text

    raise RuntimeError(f"Unknown STT provider: {provider}")


# ---------------------------------------------------------
# VOICE  —  TEXT TO SPEECH  (TTS)
# ---------------------------------------------------------

@bp.post("/api/voice/speak")
def voice_speak():
    """
    Accepts JSON: { "text": "<response text>" }
    Returns an MP3 audio stream.
    """
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="response.mp3"
        )
    except Exception as exc:
        return jsonify({"error": f"TTS failed: {exc}"}), 502


# ---------------------------------------------------------
# GENERATE REPORTS
# ---------------------------------------------------------

@bp.post("/api/report")
def report():

    try:

        data = request.get_json(force=True) or {}

        results = data.get("results", [])

        if not results:
            return jsonify({"error": "No evaluation results were provided."}), 400

        summary = summarize(results)

        json_name = save_json(results, summary)
        html_name = save_html(results, summary)
        pdf_name  = save_pdf(results, summary)

        return jsonify({
            "success": True,
            "summary": summary,
            "json_url": f"/reports/{json_name}",
            "html_url": f"/reports/{html_name}",
            "pdf_url":  f"/reports/{pdf_name}"
        })

    except Exception as exc:
        print("REPORT GENERATION ERROR:", exc)
        return jsonify({"error": f"Could not generate report: {str(exc)}"}), 500


# ---------------------------------------------------------
# SERVE GENERATED REPORT FILES
# ---------------------------------------------------------

@bp.get("/reports/<path:name>")
def report_file(name):

    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports"
    )

    return send_from_directory(reports_dir, name, as_attachment=False)
