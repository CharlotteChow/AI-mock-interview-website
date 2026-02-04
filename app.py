import html as html_lib
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

APP_TITLE = "AI Oral Interview Practice Agent"
DEFAULT_CHAT_MODEL = "gpt-5.2-codex"
YESCODE_PRESET_BASE_URL = "https://co.yes.vg/v1/responses"
YESCODE_PRESET_MODEL = "gpt-5.2-codex"
DEFAULT_API_KEY = (os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("YESCODE_API_KEY", "").strip())
DEFAULT_API_BASE_URL = (os.getenv("OPENAI_BASE_URL", "").strip() or os.getenv("YESCODE_BASE_URL", "").strip())
DEFAULT_CHAT_MODEL = (
    os.getenv("OPENAI_MODEL", "").strip()
    or os.getenv("YESCODE_MODEL", "").strip()
    or DEFAULT_CHAT_MODEL
)
WHISPER_MODEL = "whisper-1"
LOCAL_WHISPER_MODEL_OPTIONS: Tuple[str, ...] = ("base", "small", "medium")
LOCAL_WHISPER_MODEL_HINTS: Dict[str, str] = {
    "base": "更快，精度一般",
    "small": "速度/精度平衡（推荐）",
    "medium": "更准但更慢",
}

STATUS_SETUP = "Setup"
STATUS_INTERVIEWING = "Interviewing"
STATUS_REVIEW = "Review"

TIME_LIMIT_OPTIONS: Dict[str, int] = {"60s": 60, "120s": 120}
TARGET_WPM_MIN = 105
TARGET_WPM_MAX = 130
SIDEBAR_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), ".sidebar_settings.json")
CONTEXT_MEMORY_FILE = os.path.join(os.path.dirname(__file__), ".context_memory.json")

MIME_TO_EXT: Dict[str, str] = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
}

LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base").strip().lower() or "base"
if LOCAL_WHISPER_MODEL not in LOCAL_WHISPER_MODEL_OPTIONS:
    LOCAL_WHISPER_MODEL = "base"
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto").strip() or "auto"
LOCAL_WHISPER_COMPUTE_TYPE = os.getenv("LOCAL_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
DEFAULT_GATEWAY_AUTH_MODE = "both"


@dataclass(frozen=True)
class Feedback:
    score: int
    strengths: List[str]
    weaknesses: List[str]
    refined_answer: str
    perfect_answer: str
    techniques: List[str]


def _safe_json_loads(text: str) -> Any:
    """Parse JSON from an LLM response with a small amount of robustness."""
    def strip_code_fences(value: str) -> str:
        value = (value or "").strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return value

    def quote_bare_keys(value: str) -> str:
        # Turns: {hints: [...]} into {"hints": [...]} (common OpenAI-compatible formatting issue).
        return re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', value)

    def replace_single_quoted_strings(value: str) -> str:
        # Replace '...' with "..." while escaping internal double quotes.
        def repl(match: re.Match) -> str:
            inner = match.group(1)
            inner = inner.replace('"', '\\"')
            return f'"{inner}"'

        return re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", repl, value)

    def remove_trailing_commas(value: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", value)

    def normalize_py_literals(value: str) -> str:
        value = re.sub(r"\bNone\b", "null", value)
        value = re.sub(r"\bTrue\b", "true", value)
        value = re.sub(r"\bFalse\b", "false", value)
        return value

    raw = strip_code_fences(text)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt to extract the first top-level JSON object/array.
    match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
    candidate = (match.group(1) if match else raw).strip()

    for attempt in range(4):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            if attempt == 0:
                candidate = quote_bare_keys(candidate)
                continue
            if attempt == 1:
                candidate = replace_single_quoted_strings(candidate)
                candidate = normalize_py_literals(candidate)
                continue
            if attempt == 2:
                candidate = remove_trailing_commas(candidate)
                continue
            raise


def _looks_like_html(text: str) -> bool:
    t = (text or "").lstrip()
    if not t:
        return False
    if t.startswith("<!DOCTYPE") or t.startswith("<html") or t.startswith("<head") or t.startswith("<meta"):
        return True
    return bool(re.search(r"<html[\s>]|</html>|<head[\s>]|</head>|<meta[\s>]", t, flags=re.IGNORECASE))


def _is_endpoint_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg and ("endpoint not found" in msg or "not found" in msg)


def _is_invalid_client_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "403" in msg and ("invalid client" in msg or "invalid_client" in msg)


def _is_invalid_client_message(msg: str) -> bool:
    t = (msg or "").lower()
    return "403" in t and ("invalid client" in t or "invalid_client" in t)


def _normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return ""
    # Keep provider URL exactly as entered (except trailing slash trim),
    # because some OpenAI-compatible gateways do not expose /v1.
    return base_url.rstrip("/")


def _is_responses_endpoint_url(base_url: str) -> bool:
    return _normalize_base_url(base_url).lower().endswith("/responses")


def _client_base_url_for_openai_sdk(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    if normalized.lower().endswith("/responses"):
        return normalized[: -len("/responses")]
    return normalized


def _api_credential_fingerprint(api_key: str, base_url: str) -> str:
    key = (api_key or "").strip()
    if not key:
        return ""
    normalized_base = _normalize_base_url(base_url)
    payload = f"{key}|{normalized_base}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mark_current_credentials_invalid() -> None:
    ss = st.session_state
    fp = _api_credential_fingerprint(
        str(ss.get("openai_api_key_input") or ""),
        str(ss.get("api_base_url") or ""),
    )
    if fp:
        ss["api_invalid_fingerprint"] = fp


def _probe_api_connection(client, *, model: str) -> Tuple[bool, str]:
    try:
        content = _call_chat_text(
            client,
            model=(model or DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL,
            system_prompt="You are a connectivity test assistant. Keep responses brief.",
            user_prompt="Reply with exactly: OK",
            temperature=0.0,
        )
        return True, content.strip()[:120] or "OK"
    except Exception as exc:
        return False, str(exc)


def _explain_looks_invalid(explain: Any) -> bool:
    if not isinstance(explain, dict):
        return True
    for key in ("looking_for", "key_points", "framework", "keywords"):
        val = explain.get(key)
        if isinstance(val, str):
            if _looks_like_html(val):
                return True
        elif isinstance(val, list):
            sample = "\n".join([str(x) for x in val[:5]])
            if _looks_like_html(sample):
                return True
    return False


def _extract_text_from_llm_response(resp: Any) -> str:
    """
    Normalize text across OpenAI / OpenAI-compatible SDKs.

    We intentionally support multiple shapes because some providers/wrappers return
    dicts or even plain strings.
    """
    if resp is None:
        return ""

    if isinstance(resp, str):
        return resp

    if isinstance(resp, dict):
        if isinstance(resp.get("output_text"), str):
            return str(resp["output_text"])

        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0]
            if isinstance(ch0, dict):
                msg = ch0.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return str(msg["content"])
                if isinstance(ch0.get("text"), str):
                    return str(ch0["text"])
            else:
                msg = getattr(ch0, "message", None)
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return str(msg["content"])
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
                text = getattr(ch0, "text", None)
                if isinstance(text, str):
                    return text

        # Some responses-style payloads nest content more deeply.
        output = resp.get("output")
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, list) and content:
                    c0 = content[0]
                    if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                        return str(c0["text"])

        if isinstance(resp.get("text"), str):
            return str(resp["text"])

        return ""

    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    choices = getattr(resp, "choices", None)
    if isinstance(choices, list) and choices:
        ch0 = choices[0]
        if isinstance(ch0, dict):
            msg = ch0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return str(msg["content"])
            if isinstance(ch0.get("text"), str):
                return str(ch0["text"])

        msg = getattr(ch0, "message", None)
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return str(msg["content"])
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content

        text = getattr(ch0, "text", None)
        if isinstance(text, str):
            return text

    direct = getattr(resp, "content", None)
    if isinstance(direct, str):
        return direct

    text_attr = getattr(resp, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    return ""


def _truncate(text: str, max_chars: int = 12_000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n\n[...truncated...]"


def _normalize_question_text(question: str) -> str:
    q = (question or "").strip()
    q = re.sub(r"^\s*#{1,6}\s*", "", q)  # strip markdown headings
    q = re.sub(r"^\s*(?:question|q)\s*\d+\s*[:.\-]\s*", "", q, flags=re.IGNORECASE)
    q = re.sub(r"^\s*[-•]+\s*", "", q)
    return q.strip()


def _read_uploaded_bytes(obj: Any) -> bytes:
    if obj is None:
        return b""
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)

    try:
        data = obj.getvalue()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    except Exception:
        pass

    try:
        obj.seek(0)
    except Exception:
        pass

    try:
        data = obj.read()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    except Exception:
        pass

    return b""


def _audio_payload_bytes(payload: Any) -> bytes:
    if payload is None:
        return b""
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, dict):
        bts = payload.get("bytes")
        if isinstance(bts, (bytes, bytearray)):
            return bytes(bts)
    return b""


def _audio_payload_mime(payload: Any) -> str:
    if isinstance(payload, dict):
        mime = payload.get("mime")
        if isinstance(mime, str):
            return mime
    return ""


def _audio_payload_name(payload: Any) -> str:
    if isinstance(payload, dict):
        name = payload.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return "answer.wav"


def _ensure_audio_filename(name: str, mime: str) -> str:
    name = (name or "").strip() or "answer"
    ext = os.path.splitext(name)[1].lower()
    if ext:
        return name
    if mime and mime in MIME_TO_EXT:
        return name + MIME_TO_EXT[mime]
    return name + ".wav"


def _read_sidebar_settings() -> Dict[str, Any]:
    if not os.path.exists(SIDEBAR_SETTINGS_FILE):
        return {}
    try:
        with open(SIDEBAR_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_sidebar_settings(data: Dict[str, Any]) -> None:
    tmp_path = SIDEBAR_SETTINGS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, SIDEBAR_SETTINGS_FILE)


def _apply_sidebar_settings_to_session(data: Dict[str, Any]) -> None:
    ss = st.session_state

    qps = data.get("questions_per_section")
    if isinstance(qps, int):
        ss["questions_per_section"] = max(1, min(15, qps))

    tll = data.get("time_limit_label")
    if isinstance(tll, str) and tll in TIME_LIMIT_OPTIONS:
        ss["time_limit_label"] = tll

    model = data.get("chat_model")
    if isinstance(model, str) and model.strip():
        ss["chat_model"] = model.strip()

    local_whisper_model = str(data.get("local_whisper_model") or "").strip().lower()
    if local_whisper_model in LOCAL_WHISPER_MODEL_OPTIONS:
        ss["local_whisper_model"] = local_whisper_model

    base_url = data.get("api_base_url")
    if isinstance(base_url, str):
        ss["api_base_url"] = base_url.strip()

    jd = data.get("jd_text_input")
    if isinstance(jd, str):
        ss["jd_text_input"] = jd


def _read_context_memory() -> Dict[str, Any]:
    if not os.path.exists(CONTEXT_MEMORY_FILE):
        return {}
    try:
        with open(CONTEXT_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_context_memory(data: Dict[str, Any]) -> None:
    tmp_path = CONTEXT_MEMORY_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CONTEXT_MEMORY_FILE)


def _apply_context_memory_to_session(data: Dict[str, Any]) -> None:
    ss = st.session_state
    summary = data.get("summary")
    if isinstance(summary, str):
        ss["memory_summary"] = summary

    items = data.get("items")
    if isinstance(items, list):
        cleaned: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            q = str(it.get("question") or "").strip()
            t = str(it.get("transcript") or "").strip()
            r = str(it.get("refined_answer") or "").strip()
            score = it.get("score")
            try:
                score_int = int(score) if score is not None else None
            except Exception:
                score_int = None
            strengths = it.get("strengths") if isinstance(it.get("strengths"), list) else []
            weaknesses = it.get("weaknesses") if isinstance(it.get("weaknesses"), list) else []
            techniques = it.get("techniques") if isinstance(it.get("techniques"), list) else []
            cleaned.append(
                {
                    "question": q,
                    "transcript": t,
                    "refined_answer": r,
                    "score": score_int,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "techniques": techniques,
                    "ts": it.get("ts"),
                }
            )
        ss["memory_items"] = cleaned


def _on_memory_summary_editor_change() -> None:
    ss = st.session_state
    ss["memory_summary"] = str(ss.get("memory_summary_input") or "")


def _clip(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)] + "..."


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def _clean_transcript_for_scoring(transcript: str) -> str:
    text = re.sub(r"\s+", " ", (transcript or "").strip())
    # Do not let fallback placeholder text inflate the score.
    text = re.sub(r"^\s*mock transcript\s*:?\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _mock_score_from_transcript(transcript: str) -> int:
    cleaned = _clean_transcript_for_scoring(transcript)
    words = _word_count(cleaned)
    if words <= 3:
        return 1
    if words < 15:
        return 2
    if words < 35:
        return 4
    if words < 60:
        return 5
    if words < 90:
        return 6
    return 7


def _target_word_range_for_time_limit(time_limit_s: int) -> Tuple[int, int, int]:
    safe_seconds = max(30, min(300, int(time_limit_s or 60)))
    min_words = max(60, int(round(safe_seconds * TARGET_WPM_MIN / 60)))
    max_words = max(min_words + 10, int(round(safe_seconds * TARGET_WPM_MAX / 60)))
    target_words = int(round((min_words + max_words) / 2))
    return min_words, max_words, target_words


def _trim_to_max_words(text: str, max_words: int) -> str:
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip()


def _pad_to_min_words(text: str, min_words: int) -> str:
    out = (text or "").strip()
    if not out:
        out = "I answer directly, explain specific actions, and close with measurable impact."
    if out and out[-1] not in ".!?":
        out += "."
    pad_sentence = " I tie each action to business impact, explain trade-offs, and close with concrete results."
    while _word_count(out) < min_words:
        out += pad_sentence
    return out.strip()


def _tokens_with_whitespace(text: str) -> List[str]:
    # Keep whitespace tokens so rendered highlighting preserves original formatting.
    return [tok for tok in re.split(r"(\s+)", text or "") if tok != ""]


def _highlight_refined_answer(transcript: str, refined_answer: str) -> str:
    transcript_tokens = _tokens_with_whitespace(transcript)
    refined_tokens = _tokens_with_whitespace(refined_answer)

    transcript_cmp = [tok.lower() if not tok.isspace() else tok for tok in transcript_tokens]
    refined_cmp = [tok.lower() if not tok.isspace() else tok for tok in refined_tokens]
    matcher = difflib.SequenceMatcher(a=transcript_cmp, b=refined_cmp)

    out_parts: List[str] = []
    for op, _a0, _a1, b0, b1 in matcher.get_opcodes():
        if b0 >= b1:
            continue
        seg = html_lib.escape("".join(refined_tokens[b0:b1]))
        if op in {"replace", "insert"} and seg.strip():
            out_parts.append(
                "<mark style='background:#fde68a;padding:0 .1rem;border-radius:0.2rem;'>"
                + seg
                + "</mark>"
            )
        else:
            out_parts.append(seg)
    return "".join(out_parts)


def _render_refined_answer_html(transcript: str, refined_answer: str) -> str:
    rendered = _highlight_refined_answer(transcript, refined_answer)
    if not rendered.strip():
        rendered = html_lib.escape(refined_answer or "—")
    return (
        "<div style='white-space: pre-wrap; line-height:1.65;'>"
        + rendered
        + "</div>"
        + "<div style='margin-top:0.35rem;color:#64748b;font-size:0.84rem;'>"
        "高亮部分 = 相比你的 Transcript 新增/改写的优化表达。"
        "</div>"
    )


def _render_perfect_answer_html(perfect_answer: str) -> str:
    text = (perfect_answer or "").strip()
    if not text:
        return "<div style='white-space: pre-wrap; line-height:1.65;'>—</div>"

    # Prefer STAR-like sections when present; otherwise split into readable short paragraphs.
    normalized = re.sub(r"\n{3,}", "\n\n", text)
    for label in ("Situation/Task:", "Action:", "Result:"):
        normalized = re.sub(rf"\s*{re.escape(label)}\s*", f"\n\n{label} ", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip()

    if "\n\n" in normalized:
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    else:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
        paragraphs = []
        for i in range(0, len(sentences), 2):
            paragraphs.append(" ".join(sentences[i:i + 2]).strip())

    safe_html = "".join(
        f"<p style='margin:0 0 .65rem 0;'>{html_lib.escape(p)}</p>"
        for p in paragraphs
        if p
    )
    return "<div style='line-height:1.7;'>" + (safe_html or html_lib.escape(text)) + "</div>"


def _build_mock_refined_answer_from_transcript(question: str, transcript: str) -> str:
    cleaned = re.sub(r"\s+", " ", (transcript or "").strip())
    if not cleaned:
        return (
            "Situation/Task: I first clarify the problem and the success criteria.\n"
            "Action: I explain the exact steps I took, the trade-offs I made, and why.\n"
            "Result: I close with measurable impact and what I learned."
        )

    words = cleaned.split()
    if len(words) > 170:
        cleaned = " ".join(words[:170])
        words = cleaned.split()

    s_part = " ".join(words[: min(len(words), 30)]).strip()
    a_part = " ".join(words[min(len(words), 30): min(len(words), 115)]).strip()
    r_part = " ".join(words[min(len(words), 115):]).strip()

    if not a_part:
        a_part = s_part
    if not r_part:
        r_part = "This improved clarity, execution speed, and stakeholder confidence."

    question_prefix = f"For \"{question}\", " if question else ""
    return (
        f"{question_prefix}my concise answer is:\n\n"
        f"Situation/Task: {s_part}.\n"
        f"Action: {a_part}. I prioritized the highest-impact actions, aligned stakeholders early, and made trade-offs explicit.\n"
        f"Result: {r_part}. This demonstrates ownership, structured execution, and business impact."
    )


def _first_nonempty_line(text: str, max_chars: int = 120) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return _clip(line, max_chars)
    return ""


def _build_mock_perfect_answer(
    *,
    question: str,
    resume_text: str,
    jd_text: str,
    time_limit_s: int,
) -> str:
    min_words, max_words, target_words = _target_word_range_for_time_limit(time_limit_s)
    role_hint = _first_nonempty_line(jd_text, 100) or "this role"
    resume_hint = _first_nonempty_line(resume_text, 120) or "my recent relevant experience"
    q = _normalize_question_text(question) or "this interview question"

    sentence_bank = [
        f"To answer \"{q}\", I start with the business outcome that matters most for {role_hint}.",
        f"From my background in {resume_hint}, I focus on projects where I owned delivery from scoping to measurable results.",
        "Situation/Task: the team faced a clear performance and execution gap, and I was responsible for improving both reliability and delivery speed.",
        "I defined success metrics early, aligned with stakeholders on priorities, and translated ambiguous goals into a practical roadmap with milestones.",
        "Action: I broke the work into high-impact phases, handled the riskiest dependencies first, and created a communication rhythm across engineering and business partners.",
        "I made trade-offs explicit, choosing maintainable solutions that shipped quickly while preserving room for future scale and quality improvements.",
        "I tracked execution with weekly metrics, removed blockers quickly, and coached teammates so implementation quality stayed consistent under timeline pressure.",
        "Result: we improved key performance metrics, reduced avoidable rework, and delivered a solution that users and stakeholders adopted with confidence.",
        "Beyond delivery, I documented decisions, captured lessons learned, and reused the playbook so later projects started faster and with fewer risks.",
        f"What makes this relevant to {role_hint} is that I combine structured execution, clear communication, and measurable impact instead of only technical output.",
        "If I joined this team, I would apply the same approach: clarify success criteria early, prioritize highest-leverage actions, and report outcomes with concrete numbers.",
    ]

    parts: List[str] = []
    for sentence in sentence_bank:
        parts.append(sentence)
        if _word_count(" ".join(parts)) >= target_words:
            break

    perfect = " ".join(parts).strip()
    if _word_count(perfect) < min_words:
        perfect = _pad_to_min_words(perfect, min_words)
    if _word_count(perfect) > max_words:
        perfect = _trim_to_max_words(perfect, max_words)
    return perfect


def _build_memory_context_for_prompt(*, max_items: int = 3, max_chars: int = 3500) -> str:
    ss = st.session_state
    if not ss.get("use_memory_in_prompts", True):
        return ""

    summary = (ss.get("memory_summary") or "").strip()
    items: List[Dict[str, Any]] = ss.get("memory_items", []) or []
    recent = items[-max_items:] if items else []

    parts: List[str] = []
    if summary:
        parts.append("Cumulative interview context summary:\n" + _clip(summary, 1400))
    if recent:
        parts.append("Recent Q/A context:")
        for i, it in enumerate(recent, start=1):
            q = _clip(str(it.get("question") or ""), 220)
            a = _clip(str(it.get("transcript") or ""), 360)
            r = _clip(str(it.get("refined_answer") or ""), 320)
            parts.append(f"{i}) Q: {q}\n   A: {a}\n   Refined: {r}")

    context = "\n".join([p for p in parts if p.strip()]).strip()
    return _clip(context, max_chars)


def _extract_resume_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    filename = (uploaded_file.name or "").lower()
    raw = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:  # pragma: no cover
            st.sidebar.error(f"Missing dependency for PDFs: pypdf ({exc})")
            return ""

        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages).strip()
        except Exception as exc:
            st.sidebar.error(f"Could not read PDF resume: {exc}")
            return ""

    try:
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _extract_resume_texts(uploaded_files: List[Any]) -> str:
    if not uploaded_files:
        return ""

    parts: List[str] = []
    for f in uploaded_files:
        text = _extract_resume_text(f)
        name = getattr(f, "name", "") or "resume"
        if text.strip():
            parts.append(f"===== {name} =====\n{text.strip()}")

    return "\n\n".join(parts).strip()


def _get_openai_client(api_key: str, *, base_url: str = "", gateway_auth_mode: str = "authorization"):
    api_key = (api_key or "").strip()
    if not api_key:
        return None
    base_url = (base_url or "").strip()
    force_responses_api = _is_responses_endpoint_url(base_url)
    client_base_url = _client_base_url_for_openai_sdk(base_url)
    auth_mode = (gateway_auth_mode or "authorization").strip().lower()
    if auth_mode not in {"authorization", "both"}:
        auth_mode = "authorization"
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        st.sidebar.warning(f"OpenAI SDK not available; using mock mode. ({exc})")
        return None
    if client_base_url:
        default_headers: Dict[str, str] = {}
        if auth_mode == "both":
            # Some OpenAI-compatible gateways require X-API-Key in addition to Authorization.
            default_headers["X-API-Key"] = api_key
        client = OpenAI(api_key=api_key, base_url=client_base_url, default_headers=default_headers or None)
    else:
        client = OpenAI(api_key=api_key)
    if force_responses_api:
        setattr(client, "_force_responses_api", True)
    return client


def _mock_questions(resume_text: str, jd_text: str, n: int) -> List[str]:
    role_hint = "this role"
    if jd_text.strip():
        first_line = jd_text.strip().splitlines()[0].strip()
        if 0 < len(first_line) <= 80:
            role_hint = first_line

    base = [
        f"Walk me through your background and why you're a strong fit for {role_hint}.",
        "Tell me about a project you're most proud of—what was your impact and how did you measure success?",
        "Describe a time you faced a difficult problem. How did you diagnose it and what did you do?",
        "What are your strongest technical skills relevant to this position? Give concrete examples.",
        "Tell me about a time you received critical feedback. What did you change afterward?",
        "How do you prioritize when you have multiple deadlines? Describe your process.",
        "Explain a complex concept from your work to a non-technical stakeholder.",
        "What trade-offs did you make in a recent design/architecture decision? Would you do it differently now?",
    ]
    if resume_text.strip():
        base.insert(1, "Pick one item from your resume and go deep: context, challenges, actions, and results.")
    return [_normalize_question_text(q) for q in base[: max(1, n)]]


def _call_chat_text(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> Any:
    if client is None:
        raise RuntimeError("No OpenAI client configured")

    resp = None
    force_responses_api = bool(getattr(client, "_force_responses_api", False))
    if force_responses_api and hasattr(client, "responses"):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
    elif hasattr(client, "chat") and hasattr(getattr(client, "chat"), "completions"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Prefer enforced JSON when supported, but fall back for OpenAI-compatible providers.
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
    elif hasattr(client, "responses"):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
    else:
        raise RuntimeError("OpenAI client does not support chat or responses APIs")

    content = _extract_text_from_llm_response(resp).strip()
    if not content:
        raise RuntimeError("Empty LLM response content")
    if _looks_like_html(content):
        raise RuntimeError("LLM response looks like HTML (check API base URL / auth / model)")
    return content


def _call_chat_json(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> Any:
    content = _call_chat_text(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    return _safe_json_loads(content)


def generate_next_question(
    client,
    *,
    resume_text: str,
    jd_text: str,
    asked_questions: List[str],
    memory_context: str,
    model: str,
) -> str:
    asked_questions = [str(q).strip() for q in (asked_questions or []) if str(q).strip()]

    if client is None:
        mocks = _mock_questions(resume_text, jd_text, max(1, len(asked_questions) + 1))
        idx = min(len(asked_questions), len(mocks) - 1)
        return _normalize_question_text(mocks[idx])

    system_prompt = (
        "You are a strict but helpful job interviewer.\n"
        "You ONLY ask questions that are directly grounded in the candidate's resume and the job description.\n"
        "Questions must be specific, measurable, and aimed at uncovering impact, trade-offs, and depth.\n"
        "Do NOT repeat questions already asked.\n"
        "Return ONLY valid JSON."
    )

    asked_json = json.dumps(asked_questions[-20:], ensure_ascii=False, indent=2)
    mem_block = f"\n\nPrevious interview context:\n{_truncate(memory_context, 3500)}" if memory_context.strip() else ""

    user_prompt = (
        f"Resume:\n{_truncate(resume_text)}\n\n"
        f"Job description:\n{_truncate(jd_text)}"
        f"{mem_block}\n\n"
        f"Already asked questions (do not repeat):\n{asked_json}\n\n"
        "Generate ONE next interview question.\n"
        "Make it meaningfully different from the asked list, and if possible, build on the candidate's prior answers.\n"
        'Return JSON as: {"question": "..."}'
    )

    data = _call_chat_json(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.4,
    )

    question = ""
    if isinstance(data, dict):
        question = str(data.get("question") or "").strip()
    elif isinstance(data, str):
        question = data.strip()

    question = _normalize_question_text(question)
    if not question:
        # Last resort fallback
        fallback = _mock_questions(resume_text, jd_text, max(1, len(asked_questions) + 1))
        return _normalize_question_text(fallback[min(len(asked_questions), len(fallback) - 1)])

    asked_norm = { _normalize_question_text(q).lower() for q in asked_questions }
    if question.lower() in asked_norm:
        # Avoid blocking the user; just return a different mock question.
        fallback = _mock_questions(resume_text, jd_text, max(1, len(asked_questions) + 1))
        for q in fallback:
            qn = _normalize_question_text(q)
            if qn.lower() not in asked_norm:
                return qn

    return question


def generate_questions(
    client,
    *,
    resume_text: str,
    jd_text: str,
    n: int,
    model: str,
) -> List[str]:
    if client is None:
        return _mock_questions(resume_text, jd_text, n)

    system_prompt = (
        "You are a strict but helpful job interviewer.\n"
        "You ONLY ask questions that are directly grounded in the candidate's resume and the job description.\n"
        "Questions must be specific, measurable, and aimed at uncovering impact, trade-offs, and depth.\n"
        "Return ONLY valid JSON."
    )
    user_prompt = (
        f"Resume:\n{_truncate(resume_text)}\n\n"
        f"Job description:\n{_truncate(jd_text)}\n\n"
        f"Generate exactly {n} interview questions.\n"
        'Return JSON as: {"questions": ["...","..."]}'
    )

    data = _call_chat_json(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.4,
    )

    if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
        questions = [str(q).strip() for q in data["questions"] if str(q).strip()]
    elif isinstance(data, list):
        questions = [str(q).strip() for q in data if str(q).strip()]
    else:
        questions = []

    cleaned = [_normalize_question_text(q) for q in questions[:n]]
    return cleaned if cleaned else _mock_questions(resume_text, jd_text, n)


def generate_hint(
    client,
    *,
    resume_text: str,
    jd_text: str,
    question: str,
    model: str,
) -> List[str]:
    if client is None:
        return ["Use STAR (Situation/Task/Action/Result)", "Quantify impact", "Mention relevant tools/skills"]

    system_prompt = (
        "You are an interview coach.\n"
        "Given a question, provide exactly 3 short hint bullets (keywords), grounded in the resume/JD.\n"
        "Return ONLY valid JSON."
    )
    user_prompt = (
        f"Resume:\n{_truncate(resume_text)}\n\n"
        f"Job description:\n{_truncate(jd_text)}\n\n"
        f"Question:\n{question}\n\n"
        'Return JSON as: {"hints": ["keyword1","keyword2","keyword3"]}'
    )
    try:
        content = _call_chat_text(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
        )
        try:
            data = _safe_json_loads(content)
        except Exception:
            data = None

        hints: List[str] = []
        if isinstance(data, dict) and isinstance(data.get("hints"), list):
            hints = [str(x).strip() for x in data["hints"] if str(x).strip()]
        elif isinstance(data, list):
            hints = [str(x).strip() for x in data if str(x).strip()]
        else:
            # Heuristic fallback: parse first 3 non-empty lines/bullets.
            lines = []
            for line in content.splitlines():
                t = line.strip().lstrip("-•").strip()
                if t:
                    lines.append(t)
            hints = lines[:3]

        return hints[:3] if hints else ["Clarify goal", "Explain approach", "Share result"]
    except Exception:
        return ["Clarify goal", "Explain approach", "Share result"]


def _mock_question_explain(question: str) -> Dict[str, Any]:
    return {
        "looking_for": [
            "Clear communication and structured thinking",
            "Evidence of relevant experience and impact",
            "Alignment with the role requirements",
        ],
        "key_points": [
            "Give a concrete example (STAR)",
            "Quantify outcomes where possible",
            "Mention tools/skills relevant to the JD",
        ],
        "framework": [
            "1) Context: set the scene in 1–2 sentences",
            "2) Action: what you did and why (trade-offs)",
            "3) Result: metrics/impact + what you learned",
        ],
        "keywords": ["STAR", "impact", "trade-offs", "metrics", "role alignment"],
        "note": "Mock mode explanation",
    }


def _normalize_question_explain_payload(question: str, data: Any) -> Dict[str, Any]:
    base = _mock_question_explain(question)
    if not isinstance(data, dict):
        return base

    def as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            parts = [p.strip().lstrip("-•").strip() for p in value.splitlines() if p.strip()]
            return parts
        return []

    looking_for = as_list(data.get("looking_for"))[:5] or base["looking_for"]
    key_points = as_list(data.get("key_points"))[:8] or base["key_points"]
    framework = as_list(data.get("framework"))[:8] or base["framework"]
    keywords = as_list(data.get("keywords"))[:12] or base["keywords"]

    return {
        "looking_for": looking_for,
        "key_points": key_points,
        "framework": framework,
        "keywords": keywords,
    }


def generate_next_question_with_explain(
    client,
    *,
    resume_text: str,
    jd_text: str,
    asked_questions: List[str],
    memory_context: str,
    model: str,
) -> Tuple[str, Dict[str, Any]]:
    asked_questions = [str(q).strip() for q in (asked_questions or []) if str(q).strip()]

    if client is None:
        mocks = _mock_questions(resume_text, jd_text, max(1, len(asked_questions) + 1))
        idx = min(len(asked_questions), len(mocks) - 1)
        question = _normalize_question_text(mocks[idx])
        return question, _mock_question_explain(question)

    system_prompt = (
        "You are a strict but helpful interviewer and interview coach.\n"
        "You ONLY ask questions that are directly grounded in the candidate's resume and the job description.\n"
        "Questions must be specific, measurable, and aimed at uncovering impact, trade-offs, and depth.\n"
        "Do NOT repeat questions already asked.\n"
        "Also provide concise guidance for the generated question.\n"
        "Return ONLY valid JSON."
    )

    asked_json = json.dumps(asked_questions[-20:], ensure_ascii=False, indent=2)
    mem_block = f"\n\nPrevious interview context:\n{_truncate(memory_context, 3500)}" if memory_context.strip() else ""

    user_prompt = (
        f"Resume:\n{_truncate(resume_text)}\n\n"
        f"Job description:\n{_truncate(jd_text)}"
        f"{mem_block}\n\n"
        f"Already asked questions (do not repeat):\n{asked_json}\n\n"
        "Generate ONE next interview question.\n"
        "Then provide guidance for this exact question.\n"
        "Return JSON with this schema exactly:\n"
        '{"question":"...","explain":{"looking_for":[...],"key_points":[...],"framework":[...],"keywords":[...]}}'
    )

    data = _call_chat_json(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    )

    question = ""
    explain_raw: Any = None
    if isinstance(data, dict):
        question = str(data.get("question") or "").strip()
        explain_raw = data.get("explain")
    elif isinstance(data, str):
        question = data.strip()

    question = _normalize_question_text(question)
    asked_norm = {_normalize_question_text(q).lower() for q in asked_questions}
    if not question or question.lower() in asked_norm:
        fallback = _mock_questions(resume_text, jd_text, max(1, len(asked_questions) + 1))
        question = _normalize_question_text(fallback[min(len(asked_questions), len(fallback) - 1)])
        for q in fallback:
            qn = _normalize_question_text(q)
            if qn.lower() not in asked_norm:
                question = qn
                break

    explain = _normalize_question_explain_payload(question, explain_raw)
    if _explain_looks_invalid(explain):
        explain = _mock_question_explain(question)
    return question, explain


def generate_question_explain(
    client,
    *,
    resume_text: str,
    jd_text: str,
    question: str,
    memory_context: str,
    model: str,
) -> Dict[str, Any]:
    if client is None:
        return _mock_question_explain(question)

    system_prompt = (
        "You are an interview coach.\n"
        "For the given interview question, explain what the interviewer is evaluating and how to answer well.\n"
        "Ground your guidance in the resume and job description.\n"
        "You may reference the candidate's previous answers to avoid repeating generic advice.\n"
        "Return ONLY valid JSON."
    )
    mem_block = f"\n\nPrevious interview context:\n{_truncate(memory_context, 3500)}" if memory_context.strip() else ""
    user_prompt = (
        f"Resume:\n{_truncate(resume_text)}\n\n"
        f"Job description:\n{_truncate(jd_text)}"
        f"{mem_block}\n\n"
        f"Question:\n{question}\n\n"
        "Return JSON with:\n"
        '- "looking_for": 3 bullets (what interviewer is looking for)\n'
        '- "key_points": 3-6 bullets (key points to address)\n'
        '- "framework": 3-6 bullets (answer structure/framework)\n'
        '- "keywords": 5-10 short keywords/phrases\n'
        'Schema: {"looking_for": [...], "key_points": [...], "framework": [...], "keywords": [...]}\n'
    )

    try:
        data = _call_chat_json(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
        )
    except Exception:
        # Fallback: accept any text and derive bullets heuristically.
        try:
            content = _call_chat_text(
                client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
            )
        except Exception:
            return _mock_question_explain(question)

        if _looks_like_html(content):
            return _mock_question_explain(question)

        lines = [ln.strip().lstrip("-•").strip() for ln in content.splitlines() if ln.strip()]
        return {
            "looking_for": lines[:3] or _mock_question_explain(question)["looking_for"],
            "key_points": lines[3:8] or _mock_question_explain(question)["key_points"],
            "framework": lines[8:12] or _mock_question_explain(question)["framework"],
            "keywords": _mock_question_explain(question)["keywords"],
        }

    return _normalize_question_explain_payload(question, data)


@st.cache_resource(show_spinner=False)
def _load_local_whisper_model_cached(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install faster-whisper to enable local transcription fallback.") from exc

    # Keep defaults lightweight for laptop CPU usage.
    kwargs: Dict[str, Any] = {"device": device}
    if compute_type:
        kwargs["compute_type"] = compute_type

    try:
        return WhisperModel(model_name, **kwargs)
    except Exception:
        # Retry without compute_type for broader compatibility.
        kwargs.pop("compute_type", None)
        return WhisperModel(model_name, **kwargs)


def _get_local_whisper_model():
    model_name = str(st.session_state.get("local_whisper_model") or LOCAL_WHISPER_MODEL).strip().lower()
    if model_name not in LOCAL_WHISPER_MODEL_OPTIONS:
        model_name = "base"
    return _load_local_whisper_model_cached(
        model_name,
        LOCAL_WHISPER_DEVICE,
        LOCAL_WHISPER_COMPUTE_TYPE,
    )


def _transcribe_audio_local_whisper(audio_bytes: bytes, *, filename: str = "answer.wav") -> str:
    suffix = os.path.splitext((filename or "").strip())[1] or ".wav"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        model = _get_local_whisper_model()
        segments, _info = model.transcribe(
            tmp_path,
            vad_filter=False,
            beam_size=1,
            condition_on_previous_text=False,
        )
        parts = [str(getattr(seg, "text", "") or "").strip() for seg in segments]
        text = " ".join([p for p in parts if p]).strip()
        if not text:
            raise RuntimeError("Local Whisper returned an empty transcript.")
        return text
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def transcribe_audio(client, *, audio_bytes: bytes, filename: str = "answer.wav") -> str:
    if not audio_bytes:
        return ""
    ss = st.session_state
    provider_error: Optional[Exception] = None
    provider_supported = ss.get("transcription_provider_supported", None)
    if client is not None and provider_supported is not False:
        # Try provider-side transcription first.
        candidate_models = [WHISPER_MODEL, "gpt-4o-mini-transcribe", "gpt-4o-transcribe"]
        seen = set()
        for model_name in candidate_models:
            if model_name in seen:
                continue
            seen.add(model_name)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = filename  # type: ignore[attr-defined]
            try:
                resp = client.audio.transcriptions.create(model=model_name, file=audio_file)
                if isinstance(resp, str):
                    ss["transcription_provider_supported"] = True
                    return resp.strip()
                if isinstance(resp, dict):
                    ss["transcription_provider_supported"] = True
                    return str(resp.get("text") or "").strip()
                text = (getattr(resp, "text", "") or "").strip()
                if text:
                    ss["transcription_provider_supported"] = True
                    return text
            except Exception as exc:
                provider_error = exc
                msg = str(exc).lower()
                if _is_endpoint_not_found_error(exc):
                    ss["transcription_provider_supported"] = False
                    break
                if (
                    "model" in msg and ("not found" in msg or "unsupported" in msg or "does not exist" in msg)
                ) or "invalid model" in msg:
                    continue
                break

    # Fallback: local Whisper (faster-whisper).
    try:
        return _transcribe_audio_local_whisper(audio_bytes, filename=filename)
    except Exception as local_exc:
        if provider_error is not None:
            raise RuntimeError(
                f"Transcription failed on provider ({provider_error}) and local Whisper ({local_exc})."
            ) from local_exc
        if client is None:
            # In mock mode, keep an offline fallback path for UI testing.
            return "Mock transcript: I approached the problem by clarifying requirements, proposing a plan, and delivering measurable impact."
        raise RuntimeError(f"Local Whisper transcription failed: {local_exc}") from local_exc


def generate_feedback(
    client,
    *,
    resume_text: str,
    jd_text: str,
    question: str,
    transcript: str,
    memory_context: str,
    model: str,
    time_limit_s: int,
) -> Feedback:
    min_words, max_words, _target_words = _target_word_range_for_time_limit(time_limit_s)
    if client is None:
        cleaned = _clean_transcript_for_scoring(transcript)
        words = _word_count(cleaned)
        score = _mock_score_from_transcript(cleaned)
        strengths: List[str] = []
        weaknesses: List[str] = []

        if words >= 20:
            strengths.append("回答有基本结构")
        if words >= 40:
            strengths.append("信息量尚可，能支撑基本追问")
        if re.search(r"\d", cleaned):
            strengths.append("包含一定量化信息")
        if not strengths:
            strengths = ["有最基本的回应"]

        if words <= 3:
            weaknesses.append("内容几乎为空，无法评估真实能力")
        if words < 20:
            weaknesses.append("信息量不足，建议至少覆盖情境、行动、结果")
        if words < 35:
            weaknesses.append("细节不够，缺少关键决策与过程")
        if not re.search(r"\d", cleaned):
            weaknesses.append("缺少可量化结果（例如 %、时间、规模、指标）")
        weaknesses = weaknesses[:4]

        return Feedback(
            score=score,
            strengths=strengths[:4],
            weaknesses=weaknesses,
            refined_answer=_build_mock_refined_answer_from_transcript(question, transcript),
            perfect_answer=_build_mock_perfect_answer(
                question=question,
                resume_text=resume_text,
                jd_text=jd_text,
                time_limit_s=time_limit_s,
            ),
            techniques=[
                "Start with a one-sentence outcome before details.",
                "Use STAR explicitly: Situation, Task, Action, Result.",
                "Quantify scope/impact with numbers and timeline.",
                "Explain one trade-off and why your choice was right.",
            ],
        )

    system_prompt = (
        "You are a strict but helpful interviewer and communication coach.\n"
        "Evaluate the candidate's answer relative to the resume and job description.\n"
        "Use the previous interview context to keep feedback consistent and track repeated improvement areas.\n"
        "The refined sample answer MUST be a rewrite of the candidate transcript, not a generic template.\n"
        "The perfect answer MUST be grounded in the job description and resume, and represent an ideal interview answer.\n"
        "Preserve factual details from transcript/resume/JD and avoid inventing new facts.\n"
        "Be direct, specific, and actionable.\n"
        "Return ONLY valid JSON with the exact schema."
    )
    mem_block = f"\n\nPrevious interview context:\n{_truncate(memory_context, 3500)}" if memory_context.strip() else ""
    user_prompt = (
        f"Resume:\n{_truncate(resume_text, 3000)}\n\n"
        f"Job description:\n{_truncate(jd_text, 3000)}"
        f"{mem_block}\n\n"
        f"Interview question:\n{question}\n\n"
        f"Candidate transcript:\n{_truncate(transcript, 2200)}\n\n"
        "Score the answer from 1 to 10.\n"
        "List 2-4 strengths and 2-4 weaknesses.\n"
        "Provide a refined sample answer that keeps the candidate's context but is stronger in clarity, specificity, and impact.\n"
        "For refined_answer: rewrite based on the transcript's concrete content, use STAR structure, include metrics only if supported by transcript/resume/JD, and keep it around 120-220 words.\n"
        f"Provide perfect_answer: an ideal answer grounded in the resume and JD, around {min_words}-{max_words} words (normal speaking pace for {time_limit_s} seconds).\n"
        "Also provide 4-6 practical response techniques the candidate can apply next time for similar questions.\n"
        'Return JSON as: {"score": 1, "strengths": ["..."], "weaknesses": ["..."], "refined_answer": "...", "perfect_answer": "...", "techniques": ["..."]}'
    )
    data = _call_chat_json(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
    )

    try:
        score = int(data.get("score", 0))
    except Exception:
        score = 0
    strengths = [str(x).strip() for x in (data.get("strengths") or []) if str(x).strip()]
    weaknesses = [str(x).strip() for x in (data.get("weaknesses") or []) if str(x).strip()]
    refined_answer = str(data.get("refined_answer") or "").strip()
    perfect_answer = str(data.get("perfect_answer") or "").strip()
    techniques = [str(x).strip() for x in (data.get("techniques") or []) if str(x).strip()]

    if score < 1 or score > 10:
        score = 6
    if not strengths:
        strengths = ["Relevant example"]
    if not weaknesses:
        weaknesses = ["Add more specificity and metrics"]
    if not refined_answer:
        refined_answer = _build_mock_refined_answer_from_transcript(question, transcript)
    if not perfect_answer:
        perfect_answer = _build_mock_perfect_answer(
            question=question,
            resume_text=resume_text,
            jd_text=jd_text,
            time_limit_s=time_limit_s,
        )
    perfect_answer = _trim_to_max_words(_pad_to_min_words(perfect_answer, min_words), max_words)
    if not techniques:
        techniques = [
            "Open with a direct answer, then support it with one concrete story.",
            "Follow STAR and make the Action section the longest part.",
            "Add metrics (size, time, %, $, or impact on users).",
            "Close with what you learned and how it applies to this role.",
        ]

    return Feedback(
        score=score,
        strengths=strengths[:4],
        weaknesses=weaknesses[:4],
        refined_answer=refined_answer,
        perfect_answer=perfect_answer,
        techniques=techniques[:6],
    )


def _heuristic_memory_summary(items: List[Dict[str, Any]], *, max_weak: int = 5, max_str: int = 4) -> str:
    if not items:
        return ""

    strengths_all: List[str] = []
    weaknesses_all: List[str] = []
    scores: List[int] = []
    for it in items[-25:]:
        strengths_all.extend([str(x).strip() for x in (it.get("strengths") or []) if str(x).strip()])
        weaknesses_all.extend([str(x).strip() for x in (it.get("weaknesses") or []) if str(x).strip()])
        sc = it.get("score")
        if isinstance(sc, int):
            scores.append(sc)

    top_strengths = [s for s, _ in Counter(strengths_all).most_common(max_str)]
    top_weaknesses = [w for w, _ in Counter(weaknesses_all).most_common(max_weak)]
    avg_score = (sum(scores) / len(scores)) if scores else None

    lines: List[str] = []
    if avg_score is not None:
        lines.append(f"Avg score (recent): {avg_score:.1f}/10")
    if top_strengths:
        lines.append("Strengths: " + "; ".join(top_strengths))
    if top_weaknesses:
        lines.append("Improve: " + "; ".join(top_weaknesses))

    return "\n".join(lines).strip()


def update_context_memory_summary(
    client,
    *,
    previous_summary: str,
    new_turn: Dict[str, Any],
    model: str,
) -> str:
    if client is None:
        # In mock mode, keep it simple and deterministic.
        return (previous_summary or "").strip()

    system_prompt = (
        "You are an interview coach maintaining a cumulative, compact memory summary.\n"
        "This summary will be used as context for future questions and feedback.\n"
        "Constraints:\n"
        "- Keep it concise (<= 900 characters)\n"
        "- No full transcripts; extract only key signals\n"
        "- Track recurring strengths, recurring gaps, and next actions\n"
        "Return ONLY valid JSON."
    )
    user_prompt = (
        f"Previous summary:\n{_truncate(previous_summary, 1200) or '—'}\n\n"
        f"New turn:\n"
        f"- Question: {str(new_turn.get('question') or '')}\n"
        f"- Score: {str(new_turn.get('score') or '')}\n"
        f"- Strengths: {json.dumps(new_turn.get('strengths') or [], ensure_ascii=False)}\n"
        f"- Weaknesses: {json.dumps(new_turn.get('weaknesses') or [], ensure_ascii=False)}\n"
        f"- Transcript (short): {_truncate(str(new_turn.get('transcript') or ''), 500)}\n"
        f"- Refined answer (short): {_truncate(str(new_turn.get('refined_answer') or ''), 450)}\n\n"
        'Return JSON as: {"summary": "..."}'
    )
    data = _call_chat_json(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
    )
    if isinstance(data, dict) and isinstance(data.get("summary"), str) and data["summary"].strip():
        return data["summary"].strip()
    return (previous_summary or "").strip()


def append_turn_to_context_memory(
    client,
    *,
    question: str,
    transcript: str,
    feedback: Dict[str, Any],
    model: str,
) -> None:
    ss = st.session_state
    items: List[Dict[str, Any]] = ss.get("memory_items", []) or []

    new_turn = {
        "question": _normalize_question_text(question),
        "transcript": str(transcript or "").strip(),
        "refined_answer": str(feedback.get("refined_answer") or "").strip(),
        "score": int(feedback.get("score", 0)) if str(feedback.get("score", "")).strip() else None,
        "strengths": feedback.get("strengths") or [],
        "weaknesses": feedback.get("weaknesses") or [],
        "techniques": feedback.get("techniques") or [],
        "ts": time.time(),
    }

    items.append(new_turn)
    # Trim to avoid unbounded growth.
    if len(items) > 200:
        items = items[-200:]
    ss["memory_items"] = items

    # Keep memory summary local/instant to avoid an extra LLM call on every submit.
    ss["memory_summary"] = _heuristic_memory_summary(items)

    if ss.get("memory_autosave", True):
        try:
            _write_context_memory(
                {
                    "version": 1,
                    "saved_at": int(time.time()),
                    "summary": ss.get("memory_summary", "") or "",
                    "items": items,
                }
            )
        except Exception:
            pass


def _mock_section_summary(*, turns: List[Dict[str, Any]], overall_score: float) -> Dict[str, Any]:
    weaknesses: List[str] = []
    strengths: List[str] = []
    for t in turns:
        fb = t.get("feedback") or {}
        strengths.extend([str(x).strip() for x in (fb.get("strengths") or []) if str(x).strip()])
        weaknesses.extend([str(x).strip() for x in (fb.get("weaknesses") or []) if str(x).strip()])

    top_s = [s for s, _ in Counter(strengths).most_common(4)]
    top_w = [w for w, _ in Counter(weaknesses).most_common(5)]
    return {
        "summary": f"Mock section summary. Overall score: {overall_score:.1f}/10.",
        "strengths": top_s or ["Clear structure"],
        "improvements": top_w or ["Add more quantified impact"],
        "next_actions": [
            "Pick 2 stories and practice STAR delivery",
            "Add metrics + scope to each story",
            "Prepare 1–2 trade-off examples aligned to the JD",
        ],
    }


def generate_section_summary(
    client,
    *,
    turns: List[Dict[str, Any]],
    overall_score: float,
    resume_text: str,
    jd_text: str,
    model: str,
) -> Dict[str, Any]:
    if not turns:
        return {"summary": "No answers submitted yet.", "strengths": [], "improvements": [], "next_actions": []}

    if client is None:
        return _mock_section_summary(turns=turns, overall_score=overall_score)

    system_prompt = (
        "You are a strict but helpful interview coach.\n"
        "Summarize the candidate's performance for this section.\n"
        "Be specific and actionable.\n"
        "Return ONLY valid JSON."
    )

    compact_turns = []
    for t in turns[-20:]:
        fb = t.get("feedback") or {}
        compact_turns.append(
            {
                "question": str(t.get("question") or ""),
                "score": fb.get("score"),
                "strengths": fb.get("strengths") or [],
                "weaknesses": fb.get("weaknesses") or [],
                "transcript": _truncate(str(t.get("transcript") or ""), 500),
                "refined_answer": _truncate(str((fb.get("refined_answer") or "")), 420),
            }
        )

    user_prompt = (
        f"Resume (short):\n{_truncate(resume_text, 3000)}\n\n"
        f"Job description (short):\n{_truncate(jd_text, 3000)}\n\n"
        f"Overall numeric score (already computed): {overall_score:.1f}/10\n\n"
        f"Turns (JSON):\n{json.dumps(compact_turns, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with:\n"
        '- "summary": 3-6 sentences\n'
        '- "strengths": 3-6 bullets\n'
        '- "improvements": 3-6 bullets\n'
        '- "next_actions": 3-6 bullets\n'
        'Schema: {"summary": "...", "strengths": [...], "improvements": [...], "next_actions": [...]}'
    )

    data = _call_chat_json(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
    )

    if not isinstance(data, dict):
        return _mock_section_summary(turns=turns, overall_score=overall_score)

    def as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            return [v.strip().lstrip("-•").strip() for v in value.splitlines() if v.strip()]
        return []

    return {
        "summary": str(data.get("summary") or "").strip(),
        "strengths": as_list(data.get("strengths"))[:8],
        "improvements": as_list(data.get("improvements"))[:8],
        "next_actions": as_list(data.get("next_actions"))[:8],
    }


def _get_audio_payload(question_idx: int) -> Optional[Dict[str, Any]]:
    """
    Uses streamlit-audiorecorder (preferred) and returns WAV bytes.
    Falls back gracefully if the component isn't available.
    """
    def builtin_audio_input() -> Optional[Dict[str, Any]]:
        if not hasattr(st, "audio_input"):
            return None
        key = f"audio_input_{question_idx}"
        audio_file = st.audio_input("Record answer (built-in)", key=key)  # type: ignore[attr-defined]
        if audio_file is None:
            # Some Streamlit versions store the value only in session_state.
            audio_file = st.session_state.get(key)
        if audio_file is None:
            return None

        data = _read_uploaded_bytes(audio_file)
        if not data:
            return None

        mime = str(getattr(audio_file, "type", "") or "")
        name = str(getattr(audio_file, "name", "") or f"answer_{question_idx}")
        name = _ensure_audio_filename(name, mime)
        return {"bytes": data, "mime": mime, "name": name, "source": "builtin"}

    has_ffmpeg = shutil.which("ffmpeg") is not None
    has_ffprobe = shutil.which("ffprobe") is not None
    if not (has_ffmpeg and has_ffprobe):
        if not st.session_state.get("ffmpeg_notice_shown"):
            st.session_state["ffmpeg_notice_shown"] = True
            st.warning(
                "Audio recorder component needs ffmpeg/ffprobe but it's not available in PATH. "
                "Using Streamlit built-in recorder. "
                "macOS: `brew install ffmpeg`"
            )
        return builtin_audio_input()

    try:
        from audiorecorder import audiorecorder  # type: ignore
    except Exception:
        payload = builtin_audio_input()
        if payload:
            return payload
        st.info("Audio recorder component not found. Install `streamlit-audiorecorder`.")
        return None

    try:
        audio = audiorecorder("Record answer", "Recording...")
    except FileNotFoundError:
        return builtin_audio_input()

    if audio is None:
        return None

    # The component commonly returns a pydub.AudioSegment.
    try:
        if len(audio) == 0:  # type: ignore[arg-type]
            return None
    except Exception:
        pass

    try:
        buf = io.BytesIO()
        audio.export(buf, format="wav")  # type: ignore[attr-defined]
        return {"bytes": buf.getvalue(), "mime": "audio/wav", "name": f"answer_{question_idx}.wav", "source": "audiorecorder"}
    except Exception:
        # Some versions may already return raw bytes.
        if isinstance(audio, (bytes, bytearray)):
            return {"bytes": bytes(audio), "mime": "audio/wav", "name": f"answer_{question_idx}.wav", "source": "audiorecorder"}
        return None


def _init_session_state() -> None:
    ss = st.session_state
    ss.setdefault("status", STATUS_SETUP)
    ss.setdefault("questions", [])
    ss.setdefault("target_question_count", 0)
    ss.setdefault("current_idx", 0)
    ss.setdefault("resume_text", "")
    ss.setdefault("jd_text", "")
    ss.setdefault("questions_per_section", 5)
    ss.setdefault("time_limit_label", "60s")
    ss.setdefault("jd_text_input", "")
    ss.setdefault("api_base_url", DEFAULT_API_BASE_URL)
    ss.setdefault("timer_question_idx", None)
    ss.setdefault("timer_started_at", None)
    ss.setdefault("question_explain", {})  # idx -> dict
    ss.setdefault("draft_audio", {})  # idx -> audio payload (not yet submitted)
    ss.setdefault("submitted_audio", {})  # idx -> audio payload
    ss.setdefault("transcripts", {})  # idx -> text
    ss.setdefault("feedback", {})  # idx -> dict
    ss.setdefault("feedback_history", [])  # list[dict]
    ss.setdefault("section_summary", None)  # dict
    ss.setdefault("chat_model", DEFAULT_CHAT_MODEL)
    ss.setdefault("local_whisper_model", LOCAL_WHISPER_MODEL)
    ss.setdefault("ffmpeg_notice_shown", False)
    ss.setdefault("transcription_provider_supported", None)  # None/True/False
    ss.setdefault("api_invalid_fingerprint", "")

    # Cumulative context memory (persists across sections)
    ss.setdefault("use_memory_in_prompts", True)
    ss.setdefault("memory_autosave", True)
    ss.setdefault("memory_items", [])  # list[dict]
    ss.setdefault("memory_summary", "")
    ss.setdefault("memory_summary_input", ss.get("memory_summary", ""))
    ss.setdefault("context_memory_loaded", False)
    ss.setdefault("context_memory_last_action", "")

    ss.setdefault("sidebar_settings_loaded", False)
    ss.setdefault("sidebar_settings_last_action", "")

    if not ss.get("sidebar_settings_loaded"):
        data = _read_sidebar_settings()
        if data:
            _apply_sidebar_settings_to_session(data)
            ss["sidebar_settings_last_action"] = "Loaded saved sidebar settings."
        ss["sidebar_settings_loaded"] = True

    if not ss.get("context_memory_loaded"):
        data = _read_context_memory()
        if data:
            _apply_context_memory_to_session(data)
            ss["context_memory_last_action"] = "Loaded saved context memory."
        ss["context_memory_loaded"] = True


def _hide_streamlit_header_actions() -> None:
    # Hide Streamlit's top-right header actions (e.g., Deploy) for a cleaner app UI.
    st.markdown(
        """
        <style>
          [data-testid="stHeaderActionElements"] { display: none; }
          [data-testid="stToolbar"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _reset_section_state() -> None:
    ss = st.session_state
    ss["status"] = STATUS_SETUP
    ss["questions"] = []
    ss["target_question_count"] = 0
    ss["current_idx"] = 0
    ss["timer_question_idx"] = None
    ss["timer_started_at"] = None
    ss["question_explain"] = {}
    ss["draft_audio"] = {}
    ss["submitted_audio"] = {}
    ss["transcripts"] = {}
    ss["feedback"] = {}
    ss["feedback_history"] = []
    ss["section_summary"] = None


def _start_interview_section(*, client, resume_text: str, jd_text: str) -> None:
    ss = st.session_state
    n = int(ss.get("questions_per_section", 5))
    model = (ss.get("chat_model") or DEFAULT_CHAT_MODEL).strip()

    ss["target_question_count"] = n
    mem_context = _build_memory_context_for_prompt(max_items=4)
    question_explain: Dict[int, Dict[str, Any]] = {}
    try:
        first_question, first_explain = generate_next_question_with_explain(
            client,
            resume_text=resume_text,
            jd_text=jd_text,
            asked_questions=[],
            memory_context=mem_context,
            model=model,
        )
        questions = [first_question]
        question_explain[0] = first_explain
    except Exception as exc:
        if _is_invalid_client_error(exc):
            _mark_current_credentials_invalid()
            st.sidebar.error(
                "Question generation failed (403 Invalid client). "
                "This usually means the API key/base URL is not accepted by the provider. Falling back to mock questions."
            )
        else:
            st.sidebar.error(f"Question generation failed; falling back to mock questions. ({exc})")
        questions = _mock_questions(resume_text, jd_text, max(1, n))[:1]
        if questions:
            question_explain[0] = _mock_question_explain(questions[0])

    ss["status"] = STATUS_INTERVIEWING
    ss["questions"] = questions
    ss["current_idx"] = 0
    ss["timer_question_idx"] = None
    ss["timer_started_at"] = None
    ss["question_explain"] = question_explain
    ss["draft_audio"] = {}
    ss["submitted_audio"] = {}
    ss["transcripts"] = {}
    ss["feedback"] = {}
    ss["feedback_history"] = []
    ss["section_summary"] = None


def _render_sidebar() -> Tuple[Optional[Any], str, str]:
    st.sidebar.header("Configuration")

    last_msg = (st.session_state.get("sidebar_settings_last_action") or "").strip()
    if last_msg:
        st.sidebar.caption(last_msg)
        st.session_state["sidebar_settings_last_action"] = ""

    mem_msg = (st.session_state.get("context_memory_last_action") or "").strip()
    if mem_msg:
        st.sidebar.caption(mem_msg)
        st.session_state["context_memory_last_action"] = ""

    # Apply deferred updates before widgets are instantiated in this rerun.
    pending_sidebar_updates = st.session_state.pop("_pending_sidebar_updates", None)
    if isinstance(pending_sidebar_updates, dict):
        for k, v in pending_sidebar_updates.items():
            st.session_state[k] = v

    pending_settings_load = st.session_state.pop("_pending_sidebar_settings_load", None)
    if isinstance(pending_settings_load, dict):
        _apply_sidebar_settings_to_session(pending_settings_load)

    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=DEFAULT_API_KEY,
        key="openai_api_key_input",
        help="Leave blank to use Mock Mode. Also supports YESCODE_API_KEY env. For third-party gateways, this key is auto-sent as X-API-Key.",
    )
    base_url = st.sidebar.text_input(
        "API Base URL (optional)",
        key="api_base_url",
        help="Use this for OpenAI-compatible providers (leave blank for OpenAI). Also supports OPENAI_BASE_URL / YESCODE_BASE_URL env.",
    )
    base_url = _normalize_base_url(base_url)

    col_provider1, col_provider2 = st.sidebar.columns(2)
    use_third_party_api = col_provider1.button(
        "Use Third-Party API",
        use_container_width=True,
        help=f"Set Base URL to {YESCODE_PRESET_BASE_URL} and model to {YESCODE_PRESET_MODEL}.",
    )
    use_openai = col_provider2.button(
        "Use OpenAI Default",
        use_container_width=True,
        help="Clear Base URL and keep model editable manually.",
    )
    if use_third_party_api:
        st.session_state["_pending_sidebar_updates"] = {
            "api_base_url": YESCODE_PRESET_BASE_URL,
            "chat_model": os.getenv("YESCODE_MODEL", "").strip() or YESCODE_PRESET_MODEL,
        }
        st.session_state["sidebar_settings_last_action"] = "Applied third-party API preset."
        st.rerun()
    if use_openai:
        st.session_state["_pending_sidebar_updates"] = {"api_base_url": ""}
        st.session_state["sidebar_settings_last_action"] = "Switched to OpenAI default base URL."
        st.rerun()

    ss = st.session_state
    current_fp = _api_credential_fingerprint(api_key, base_url)
    saved_invalid_fp = str(ss.get("api_invalid_fingerprint") or "")
    if saved_invalid_fp and current_fp and saved_invalid_fp != current_fp:
        # Credentials changed; clear previous invalid marker.
        ss["api_invalid_fingerprint"] = ""
        saved_invalid_fp = ""

    mode_reason = ""
    if saved_invalid_fp and current_fp and saved_invalid_fp == current_fp:
        client = None
        mode_reason = "当前这组 API 凭证曾返回 403 Invalid client"
        st.sidebar.warning(
            "Detected 403 Invalid client for the current API key/base URL. "
            "Using Mock Mode until credentials change."
        )
    else:
        client = _get_openai_client(
            api_key,
            base_url=base_url,
            gateway_auth_mode=DEFAULT_GATEWAY_AUTH_MODE,
        )
        if client is None and api_key.strip():
            mode_reason = "API 客户端初始化失败（请检查 SDK / Base URL / Key）"

    if client is None and not api_key.strip():
        mode_reason = "未填写 API Key"
        st.sidebar.info("Mock Mode: No API key detected. Questions/feedback will be simulated.")

    runtime_mode = "online" if client is not None else "mock"
    st.session_state["runtime_mode"] = runtime_mode
    st.session_state["runtime_mode_reason"] = mode_reason

    st.sidebar.divider()
    if runtime_mode == "online":
        st.sidebar.success("Current Mode: Online（将调用真实 API）")
    else:
        detail = f"（{mode_reason}）" if mode_reason else ""
        st.sidebar.warning(f"Current Mode: Mock{detail}")

    if base_url:
        st.sidebar.caption(f"Base URL in use: {base_url}")
    else:
        st.sidebar.caption("Base URL in use: OpenAI default")

    col_mo1, col_mo2 = st.sidebar.columns(2)
    retry_api = col_mo1.button(
        "Retry API",
        use_container_width=True,
        disabled=not api_key.strip(),
        help="Clear the cached 403 marker and retry current credentials.",
    )
    test_api = col_mo2.button(
        "Test API",
        use_container_width=True,
        disabled=not api_key.strip(),
        help="Send a tiny test request with current key/base URL/model.",
    )

    if retry_api:
        st.session_state["api_invalid_fingerprint"] = ""
        st.session_state["sidebar_settings_last_action"] = "Cleared cached 403 marker. Please click Start Interview Section again."
        st.rerun()

    if test_api:
        probe_client = _get_openai_client(
            api_key,
            base_url=base_url,
            gateway_auth_mode=DEFAULT_GATEWAY_AUTH_MODE,
        )
        if probe_client is None:
            st.sidebar.error("Could not initialize API client with current settings.")
        else:
            with st.spinner("Testing API connectivity..."):
                ok, detail = _probe_api_connection(
                    probe_client,
                    model=str(st.session_state.get("chat_model") or DEFAULT_CHAT_MODEL).strip(),
                )
            if ok:
                st.session_state["api_invalid_fingerprint"] = ""
                st.sidebar.success(f"API test passed. Provider replied: {detail}")
            else:
                if _is_invalid_client_message(detail):
                    _mark_current_credentials_invalid()
                st.sidebar.error(f"API test failed: {detail}")

    uploaded_resumes = st.sidebar.file_uploader(
        "Upload Resume(s) (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    jd_text = st.sidebar.text_area(
        "Job Description (JD)",
        key="jd_text_input",
        height=220,
        placeholder="Paste the job description here…",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Interview Settings")
    st.sidebar.number_input(
        "Questions per Section",
        min_value=1,
        max_value=15,
        value=int(st.session_state.get("questions_per_section", 5)),
        step=1,
        key="questions_per_section",
    )
    time_limit_options = list(TIME_LIMIT_OPTIONS.keys())
    current_time_limit = st.session_state.get("time_limit_label", "60s")
    if current_time_limit not in TIME_LIMIT_OPTIONS:
        current_time_limit = "60s"
    st.sidebar.radio(
        "Time Limit per Question",
        options=time_limit_options,
        index=time_limit_options.index(current_time_limit),
        key="time_limit_label",
        horizontal=True,
    )
    st.sidebar.text_input(
        "Chat Model",
        key="chat_model",
        help="Example: gpt-4o-mini. Enter any model id available to your API/provider.",
    )
    current_local_model = str(st.session_state.get("local_whisper_model") or LOCAL_WHISPER_MODEL).strip().lower()
    if current_local_model not in LOCAL_WHISPER_MODEL_OPTIONS:
        current_local_model = "base"
    local_model_options = list(LOCAL_WHISPER_MODEL_OPTIONS)
    selected_local_model = st.sidebar.selectbox(
        "建议模型选择（本地转写）",
        options=local_model_options,
        index=local_model_options.index(current_local_model),
        format_func=lambda x: f"{x}: {LOCAL_WHISPER_MODEL_HINTS.get(x, '')}",
        help="用于本地 faster-whisper 转写 fallback；默认 base。",
    )
    st.session_state["local_whisper_model"] = selected_local_model

    st.sidebar.divider()
    st.sidebar.subheader("Saved Settings")
    include_jd = st.sidebar.checkbox(
        "Also save JD text",
        value=True,
        help="Saves the JD textarea contents to disk in the settings file.",
    )
    has_saved = os.path.exists(SIDEBAR_SETTINGS_FILE)
    col_s1, col_s2, col_s3 = st.sidebar.columns(3)
    save_settings = col_s1.button("Save", use_container_width=True, key="save_sidebar_settings")
    load_settings = col_s2.button("Load", use_container_width=True, disabled=not has_saved, key="load_sidebar_settings")
    clear_settings = col_s3.button("Clear", use_container_width=True, disabled=not has_saved, key="clear_sidebar_settings")

    if save_settings:
        try:
            payload: Dict[str, Any] = {
                "version": 1,
                "saved_at": int(time.time()),
                "questions_per_section": int(st.session_state.get("questions_per_section", 5)),
                "time_limit_label": str(st.session_state.get("time_limit_label", "60s")),
                "chat_model": str(st.session_state.get("chat_model", DEFAULT_CHAT_MODEL)).strip() or DEFAULT_CHAT_MODEL,
                "local_whisper_model": str(st.session_state.get("local_whisper_model", LOCAL_WHISPER_MODEL)).strip().lower(),
                "api_base_url": _normalize_base_url(str(st.session_state.get("api_base_url", "")).strip()),
            }
            if include_jd:
                payload["jd_text_input"] = str(st.session_state.get("jd_text_input", ""))
            _write_sidebar_settings(payload)
            st.session_state["sidebar_settings_last_action"] = f"Saved settings to {os.path.basename(SIDEBAR_SETTINGS_FILE)}."
            st.sidebar.success(st.session_state["sidebar_settings_last_action"])
        except Exception as exc:
            st.sidebar.error(f"Failed to save settings: {exc}")

    if load_settings:
        data = _read_sidebar_settings()
        if not data:
            st.sidebar.warning("No saved settings found.")
        else:
            st.session_state["_pending_sidebar_settings_load"] = data
            st.session_state["sidebar_settings_last_action"] = "Loaded saved settings."
            st.rerun()

    if clear_settings:
        try:
            os.remove(SIDEBAR_SETTINGS_FILE)
            st.session_state["sidebar_settings_last_action"] = "Cleared saved settings."
            st.rerun()
        except FileNotFoundError:
            st.session_state["sidebar_settings_last_action"] = "No saved settings file found."
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Failed to clear settings: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Context Memory")
    st.sidebar.checkbox(
        "Use accumulated context in prompts",
        key="use_memory_in_prompts",
        help="If enabled, your previous answers and improved answers are used as context for later questions and feedback.",
    )
    st.sidebar.checkbox(
        "Autosave memory to disk",
        key="memory_autosave",
        help=f"Saves to {os.path.basename(CONTEXT_MEMORY_FILE)} automatically after each analyzed answer.",
    )
    st.sidebar.caption(f"Turns stored: {len(st.session_state.get('memory_items', []) or [])}")
    # Keep the editable text area in sync with canonical memory_summary.
    st.session_state["memory_summary_input"] = str(st.session_state.get("memory_summary") or "")
    st.sidebar.text_area(
        "Memory summary (editable)",
        key="memory_summary_input",
        on_change=_on_memory_summary_editor_change,
        height=140,
        placeholder="A running summary of your interview performance and story will appear here…",
    )

    with st.sidebar.expander("Recent turns", expanded=False):
        items = st.session_state.get("memory_items", []) or []
        for it in items[-5:]:
            q = str(it.get("question") or "").strip()
            score = it.get("score")
            score_txt = f"{score}/10" if isinstance(score, int) else "—"
            st.markdown(f"- **{score_txt}** {q[:120]}{'…' if len(q) > 120 else ''}")

    has_mem_saved = os.path.exists(CONTEXT_MEMORY_FILE)
    col_m1, col_m2, col_m3 = st.sidebar.columns(3)
    save_mem = col_m1.button("Save", use_container_width=True, key="save_context_memory")
    load_mem = col_m2.button("Load", use_container_width=True, disabled=not has_mem_saved, key="load_context_memory")
    clear_mem = col_m3.button("Clear", use_container_width=True, key="clear_context_memory")

    if save_mem:
        try:
            payload = {
                "version": 1,
                "saved_at": int(time.time()),
                "summary": str(st.session_state.get("memory_summary", "") or ""),
                "items": st.session_state.get("memory_items", []) or [],
            }
            _write_context_memory(payload)
            st.session_state["context_memory_last_action"] = f"Saved context memory to {os.path.basename(CONTEXT_MEMORY_FILE)}."
            st.sidebar.success(st.session_state["context_memory_last_action"])
        except Exception as exc:
            st.sidebar.error(f"Failed to save memory: {exc}")

    if load_mem:
        data = _read_context_memory()
        if not data:
            st.sidebar.warning("No saved memory found.")
        else:
            _apply_context_memory_to_session(data)
            st.session_state["context_memory_last_action"] = "Loaded saved context memory."
            st.rerun()

    if clear_mem:
        st.session_state["memory_items"] = []
        st.session_state["memory_summary"] = ""
        try:
            if os.path.exists(CONTEXT_MEMORY_FILE):
                os.remove(CONTEXT_MEMORY_FILE)
        except Exception:
            pass
        st.session_state["context_memory_last_action"] = "Cleared memory."
        st.rerun()

    mem_json = json.dumps(
        {
            "version": 1,
            "summary": st.session_state.get("memory_summary", "") or "",
            "items": st.session_state.get("memory_items", []) or [],
        },
        ensure_ascii=False,
        indent=2,
    )
    st.sidebar.download_button(
        "Download memory JSON",
        data=mem_json,
        file_name="context_memory.json",
        mime="application/json",
        use_container_width=True,
    )
    import_file = st.sidebar.file_uploader("Import memory JSON", type=["json"], key="import_memory_file")
    if import_file is not None:
        if st.sidebar.button("Import now", use_container_width=True):
            try:
                data = _safe_json_loads(_read_uploaded_bytes(import_file).decode("utf-8", errors="ignore"))
                if isinstance(data, dict):
                    _apply_context_memory_to_session(data)
                    st.session_state["context_memory_last_action"] = "Imported context memory."
                    st.rerun()
                else:
                    st.sidebar.error("Invalid memory JSON: expected an object.")
            except Exception as exc:
                st.sidebar.error(f"Failed to import: {exc}")

    st.sidebar.divider()
    col_a, col_b = st.sidebar.columns(2)
    start_label = "Start Interview Section (Online)" if runtime_mode == "online" else "Start Interview Section (Mock)"
    start_clicked = col_a.button(start_label, type="primary", use_container_width=True)
    reset_clicked = col_b.button("Reset", use_container_width=True)
    if reset_clicked:
        _reset_section_state()

    resume_text = _extract_resume_texts(uploaded_resumes or [])
    if start_clicked:
        if not resume_text and not jd_text.strip():
            st.sidebar.warning("Add a resume and/or job description for best results (mock mode still works).")
        st.session_state["resume_text"] = resume_text
        st.session_state["jd_text"] = jd_text
        _start_interview_section(client=client, resume_text=resume_text, jd_text=jd_text)

    return client, resume_text, jd_text


def _render_timer(time_limit_s: int, question_idx: int) -> None:
    ss = st.session_state
    if ss.get("timer_question_idx") != question_idx:
        ss["timer_question_idx"] = question_idx
        ss["timer_started_at"] = None

    started_at_raw = ss.get("timer_started_at")
    if not started_at_raw:
        components.html(
            """
            <script>
              (function() {
                const k = "ai_interview_timer_interval";
                if (window[k]) { clearInterval(window[k]); window[k] = null; }
              })();
            </script>
            """,
            height=0,
        )
        st.info(f"Timer is not running. Click **Start Answer** when you begin recording. (Limit {time_limit_s}s)")
        return

    started_at = float(ss.get("timer_started_at") or time.time())
    started_ms = int(started_at * 1000)
    end_ms = started_ms + int(time_limit_s * 1000)

    components.html(
        f"""
        <div style="font-family: ui-sans-serif, system-ui, -apple-system; margin: 0.25rem 0 0.75rem 0;">
          <div id="timer-text-{question_idx}" style="font-size: 0.9rem; margin-bottom: 0.35rem;"></div>
          <div style="background: #e6e6e6; height: 10px; border-radius: 9999px; overflow: hidden;">
            <div id="timer-bar-{question_idx}" style="height: 10px; width: 0%; background: #1f77b4; transition: width 0.2s linear;"></div>
          </div>
          <div id="timer-msg-{question_idx}" style="display:none; margin-top: 0.5rem; color: #8a6d3b; background: #fcf8e3; padding: 0.5rem 0.75rem; border-radius: 0.25rem;">
            Time is up. Submit what you have, then review the feedback.
          </div>
        </div>
        <script>
          (function() {{
            const startAt = {started_ms};
            const endAt = {end_ms};
            const limitSec = {time_limit_s};
            const textEl = document.getElementById("timer-text-{question_idx}");
            const barEl = document.getElementById("timer-bar-{question_idx}");
            const msgEl = document.getElementById("timer-msg-{question_idx}");

            function tick() {{
              const now = Date.now();
              const remainingMs = Math.max(0, endAt - now);
              const remainingSec = Math.ceil(remainingMs / 1000);
              const elapsedSec = Math.min(limitSec, Math.max(0, (now - startAt) / 1000));
              const progress = Math.min(1, elapsedSec / limitSec);

              if (textEl) textEl.textContent = "Time remaining: " + remainingSec + "s (limit " + limitSec + "s)";
              if (barEl) barEl.style.width = Math.round(progress * 100) + "%";

              if (msgEl) {{
                msgEl.style.display = remainingMs <= 0 ? "block" : "none";
              }}
            }}

            tick();
            const k = "ai_interview_timer_interval";
            if (window[k]) clearInterval(window[k]);
            window[k] = setInterval(tick, 200);
          }})();
        </script>
        """,
        height=96,
    )


def _advance_to_next_question(client) -> None:
    ss = st.session_state
    ss["timer_question_idx"] = None
    ss["timer_started_at"] = None

    ss["current_idx"] += 1
    total_target = int(ss.get("target_question_count") or 0)
    if total_target <= 0:
        total_target = len(ss.get("questions", [])) or max(1, int(ss.get("questions_per_section", 5)))
        ss["target_question_count"] = total_target

    if ss["current_idx"] >= total_target:
        ss["status"] = STATUS_REVIEW
        return

    questions: List[str] = ss.get("questions", [])
    explains = ss.get("question_explain", {}) or {}
    if not isinstance(explains, dict):
        explains = {}
    if ss["current_idx"] >= len(questions):
        try:
            mem_context = _build_memory_context_for_prompt(max_items=4)
            next_idx = len(questions)
            next_q, next_explain = generate_next_question_with_explain(
                client,
                resume_text=ss.get("resume_text", ""),
                jd_text=ss.get("jd_text", ""),
                asked_questions=questions,
                memory_context=mem_context,
                model=(ss.get("chat_model") or DEFAULT_CHAT_MODEL).strip(),
            )
            questions.append(next_q)
            ss["questions"] = questions
            explains[next_idx] = next_explain
            ss["question_explain"] = explains
        except Exception:
            # Let _render_interview try again (and display an error) if needed.
            pass


def _render_interview(client) -> None:
    ss = st.session_state
    questions: List[str] = ss.get("questions", [])
    explains = ss.get("question_explain", {}) or {}
    if not isinstance(explains, dict):
        explains = {}
    if not questions:
        st.warning("No questions generated yet. Use the sidebar to start an interview section.")
        return

    total_target = int(ss.get("target_question_count") or 0)
    if total_target <= 0:
        total_target = max(1, int(ss.get("questions_per_section", 5)))
        ss["target_question_count"] = total_target

    idx = int(ss.get("current_idx", 0))
    idx = max(0, min(idx, total_target - 1))

    # Ensure the current question exists (questions are generated progressively).
    if idx >= len(questions):
        with st.spinner("Generating next question..."):
            while len(questions) <= idx and len(questions) < total_target:
                try:
                    mem_context = _build_memory_context_for_prompt(max_items=4)
                    next_idx = len(questions)
                    next_q, next_explain = generate_next_question_with_explain(
                        client,
                        resume_text=ss.get("resume_text", ""),
                        jd_text=ss.get("jd_text", ""),
                        asked_questions=questions,
                        memory_context=mem_context,
                        model=(ss.get("chat_model") or DEFAULT_CHAT_MODEL).strip(),
                    )
                    questions.append(next_q)
                    explains[next_idx] = next_explain
                except Exception as exc:
                    st.error(f"Could not generate next question: {exc}")
                    break
        ss["questions"] = questions
        ss["question_explain"] = explains

    if idx >= len(questions):
        st.error("No question available. Try restarting the section.")
        return

    question = _normalize_question_text(questions[idx])

    st.subheader(f"Question {idx + 1} / {total_target}")
    st.markdown(
        f"""
        <div style="
          font-size: 1.05rem;
          line-height: 1.6;
          padding: 0.9rem 1rem;
          border-radius: 0.75rem;
          background: rgba(31, 119, 180, 0.06);
          border: 1px solid rgba(31, 119, 180, 0.18);
        ">
          {html_lib.escape(question)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    time_limit_s = TIME_LIMIT_OPTIONS.get(ss.get("time_limit_label", "60s"), 60)

    # Question guidance (auto; no "Show Hint" click needed)
    if idx not in ss.get("question_explain", {}):
        with st.spinner("Generating question guidance..."):
            try:
                explain = generate_question_explain(
                    client,
                    resume_text=ss.get("resume_text", ""),
                    jd_text=ss.get("jd_text", ""),
                    question=question,
                    memory_context=_build_memory_context_for_prompt(max_items=4),
                    model=(ss.get("chat_model") or DEFAULT_CHAT_MODEL).strip(),
                )
            except Exception:
                explain = _mock_question_explain(question)
            ss["question_explain"][idx] = explain

    explain = ss.get("question_explain", {}).get(idx)
    if not explain or _explain_looks_invalid(explain):
        explain = _mock_question_explain(question)
        ss["question_explain"][idx] = explain

    st.markdown("### Question Explain")
    note = str(explain.get("note") or "").strip()
    if note:
        st.caption(note)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**What the interviewer is looking for**")
        looking_for = explain.get("looking_for") or []
        st.write("- " + "\n- ".join(looking_for) if looking_for else "—")
    with col_b:
        st.markdown("**Key points to address**")
        key_points = explain.get("key_points") or []
        st.write("- " + "\n- ".join(key_points) if key_points else "—")

    st.markdown("**Answer framework**")
    framework = explain.get("framework") or []
    st.write("- " + "\n- ".join(framework) if framework else "—")

    st.markdown("**Keywords**")
    keywords = explain.get("keywords") or []
    st.write(", ".join(keywords) if keywords else "—")

    st.divider()
    st.markdown("### Answer Timer")
    already_analyzed = idx in ss.get("feedback", {})
    col_t1, col_t2, col_t3, col_t4 = st.columns([1, 1, 1, 5])
    with col_t1:
        start_timer = st.button("Start Answer", use_container_width=True, disabled=already_analyzed, key=f"start_timer_{idx}")
    with col_t2:
        restart_timer = st.button("Restart", use_container_width=True, disabled=already_analyzed, key=f"restart_timer_{idx}")
    with col_t3:
        reset_timer = st.button("Reset", use_container_width=True, disabled=already_analyzed, key=f"reset_timer_{idx}")
    with col_t4:
        if start_timer or restart_timer:
            ss["timer_question_idx"] = idx
            ss["timer_started_at"] = time.time()
        elif reset_timer:
            ss["timer_question_idx"] = idx
            ss["timer_started_at"] = None
        _render_timer(time_limit_s, idx)
    st.divider()
    st.markdown("### Record your answer")
    payload = _get_audio_payload(idx)
    if payload:
        ss["draft_audio"][idx] = payload

    draft_audio = ss.get("draft_audio", {}).get(idx)
    draft_bytes = _audio_payload_bytes(draft_audio)
    if draft_bytes:
        st.caption("Playback (before submit)")
        mime = _audio_payload_mime(draft_audio) or None
        st.audio(draft_bytes, format=mime)

    if already_analyzed:
        st.info("Answer already submitted for this question. You can move to the next question.")

    col_submit, col_next = st.columns(2)
    with col_submit:
        submit_clicked = st.button(
            "Submit & Analyze",
            type="primary",
            use_container_width=True,
            disabled=already_analyzed,
        )
    with col_next:
        is_last = idx == (total_target - 1)
        next_label = "Finish Section" if is_last else "Next Question"
        next_clicked = st.button(next_label, use_container_width=True, disabled=not already_analyzed)

    if submit_clicked:
        if not draft_bytes:
            # Last-chance recovery: read from the built-in audio widget state if present.
            audio_file_obj = ss.get(f"audio_input_{idx}") or ss.get("builtin_audio_input")
            recovered = None
            if audio_file_obj is not None:
                data = _read_uploaded_bytes(audio_file_obj)
                if data:
                    mime = str(getattr(audio_file_obj, "type", "") or "")
                    name = str(getattr(audio_file_obj, "name", "") or f"answer_{idx}")
                    recovered = {"bytes": data, "mime": mime, "name": _ensure_audio_filename(name, mime), "source": "builtin"}
            if recovered:
                ss["draft_audio"][idx] = recovered
                draft_audio = recovered
                draft_bytes = _audio_payload_bytes(draft_audio)
            else:
                st.warning("Please record audio before submitting.")
                st.stop()

        ss["submitted_audio"][idx] = draft_audio
        with st.spinner("Transcribing (if needed) and generating feedback..."):
            filename = _ensure_audio_filename(_audio_payload_name(draft_audio), _audio_payload_mime(draft_audio))
            try:
                transcript = transcribe_audio(client, audio_bytes=draft_bytes, filename=filename)
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                st.info(
                    "Provider transcription failed and local Whisper fallback also failed. "
                    "Check ffmpeg and faster-whisper installation, then try again."
                )
                st.stop()

            if not transcript:
                st.error("Transcription failed (empty transcript). Please record and try again.")
                st.stop()

            mem_ctx = _build_memory_context_for_prompt(max_items=2)
            try:
                feedback = generate_feedback(
                    client,
                    resume_text=ss.get("resume_text", ""),
                    jd_text=ss.get("jd_text", ""),
                    question=question,
                    transcript=transcript,
                    memory_context=mem_ctx,
                    model=(ss.get("chat_model") or DEFAULT_CHAT_MODEL).strip(),
                    time_limit_s=time_limit_s,
                )
            except Exception as exc:
                if _is_invalid_client_error(exc):
                    _mark_current_credentials_invalid()
                    st.warning(
                        "当前 API 凭证返回 403 Invalid client，已自动切换为 transcript-based mock feedback。"
                    )
                else:
                    st.warning(f"Feedback generation failed; using mock feedback. ({exc})")
                feedback = generate_feedback(
                    None,
                    resume_text=ss.get("resume_text", ""),
                    jd_text=ss.get("jd_text", ""),
                    question=question,
                    transcript=transcript,
                    memory_context=mem_ctx,
                    model=(ss.get("chat_model") or DEFAULT_CHAT_MODEL).strip(),
                    time_limit_s=time_limit_s,
                )

            ss["transcripts"][idx] = transcript
            ss["feedback"][idx] = {
                "score": feedback.score,
                "strengths": feedback.strengths,
                "weaknesses": feedback.weaknesses,
                "refined_answer": feedback.refined_answer,
                "perfect_answer": feedback.perfect_answer,
                "time_limit_s": time_limit_s,
                "perfect_answer_word_range": list(_target_word_range_for_time_limit(time_limit_s)[:2]),
                "techniques": feedback.techniques,
            }
            ss["feedback_history"].append(
                {
                    "question_idx": idx,
                    "question": question,
                    "transcript": transcript,
                    "feedback": ss["feedback"][idx],
                    "ts": time.time(),
                }
            )
            append_turn_to_context_memory(
                client,
                question=question,
                transcript=transcript,
                feedback=ss["feedback"][idx],
                model=(ss.get("chat_model") or DEFAULT_CHAT_MODEL).strip(),
            )

    # Feedback display (immediate)
    if idx in ss.get("feedback", {}):
        st.divider()
        st.markdown("### Your Transcript")
        st.write(ss.get("transcripts", {}).get(idx, ""))

        f = ss.get("feedback", {}).get(idx, {})
        st.markdown("### AI Analysis")
        st.metric("Score", f"{int(f.get('score', 0))}/10")

        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("**Strengths**")
            strengths = f.get("strengths") or []
            st.write("- " + "\n- ".join(strengths) if strengths else "—")
        with col_w:
            st.markdown("**Weaknesses**")
            weaknesses = f.get("weaknesses") or []
            st.write("- " + "\n- ".join(weaknesses) if weaknesses else "—")

        st.markdown("**Refined Sample Answer**")
        st.markdown(
            _render_refined_answer_html(
                ss.get("transcripts", {}).get(idx, "") or "",
                f.get("refined_answer") or "",
            ),
            unsafe_allow_html=True,
        )
        perfect_answer = str(f.get("perfect_answer") or "").strip()
        pa_time_limit = int(f.get("time_limit_s") or time_limit_s)
        pa_range = f.get("perfect_answer_word_range") or []
        if isinstance(pa_range, list) and len(pa_range) >= 2:
            pa_min_words = int(pa_range[0])
            pa_max_words = int(pa_range[1])
        else:
            pa_min_words, pa_max_words, _ = _target_word_range_for_time_limit(pa_time_limit)
        st.markdown(f"**Perfect Answer (Resume + JD, ~{pa_time_limit}s)**")
        st.caption(f"Target length: {pa_min_words}-{pa_max_words} words | Current: {_word_count(perfect_answer)} words")
        st.markdown(_render_perfect_answer_html(perfect_answer), unsafe_allow_html=True)
        st.markdown("**Response Techniques**")
        techniques = f.get("techniques") or []
        st.write("- " + "\n- ".join(techniques) if techniques else "—")

    if next_clicked:
        _advance_to_next_question(client)
        st.rerun()


def _render_review(client) -> None:
    ss = st.session_state
    questions: List[str] = ss.get("questions", [])
    feedback_map: Dict[int, Dict[str, Any]] = ss.get("feedback", {})

    st.header("Section Review")
    if not questions:
        st.info("No completed section yet.")
        return

    scores = []
    rows = []
    all_weaknesses: List[str] = []
    for i, q in enumerate(questions):
        f = feedback_map.get(i)
        score = int(f.get("score", 0)) if f else 0
        if f:
            scores.append(score)
            all_weaknesses.extend([str(w).strip() for w in (f.get("weaknesses") or []) if str(w).strip()])
        rows.append({"#": i + 1, "Question": q, "Score": score})

    overall = (sum(scores) / len(scores)) if scores else 0.0
    st.metric("Overall Performance Score", f"{overall:.1f}/10")

    col_sum_a, col_sum_b = st.columns([1, 1])
    with col_sum_a:
        regen = st.button("Regenerate summary", use_container_width=True)
    with col_sum_b:
        save_to_memory = st.button("Save summary to memory", use_container_width=True)

    if regen:
        ss["section_summary"] = None

    if ss.get("section_summary") is None:
        with st.spinner("Generating section summary..."):
            try:
                ss["section_summary"] = generate_section_summary(
                    client,
                    turns=ss.get("feedback_history", []) or [],
                    overall_score=overall,
                    resume_text=ss.get("resume_text", ""),
                    jd_text=ss.get("jd_text", ""),
                    model=(ss.get("chat_model") or DEFAULT_CHAT_MODEL).strip(),
                )
            except Exception:
                ss["section_summary"] = _mock_section_summary(turns=ss.get("feedback_history", []) or [], overall_score=overall)

    section_summary = ss.get("section_summary") or {}
    st.subheader("Section Summary")
    st.write(section_summary.get("summary") or "—")

    col_s, col_i, col_n = st.columns(3)
    with col_s:
        st.markdown("**Strengths**")
        strengths = section_summary.get("strengths") or []
        st.write("- " + "\n- ".join(strengths) if strengths else "—")
    with col_i:
        st.markdown("**Improvements**")
        improvements = section_summary.get("improvements") or []
        st.write("- " + "\n- ".join(improvements) if improvements else "—")
    with col_n:
        st.markdown("**Next actions**")
        next_actions = section_summary.get("next_actions") or []
        st.write("- " + "\n- ".join(next_actions) if next_actions else "—")

    if save_to_memory and section_summary.get("summary"):
        # Append the section summary into the running memory summary (kept compact).
        current = str(ss.get("memory_summary") or "").strip()
        addition = str(section_summary.get("summary") or "").strip()
        combined = (current + "\n\nSection summary:\n" + addition).strip() if current else ("Section summary:\n" + addition)
        ss["memory_summary"] = _clip(combined, 1600)
        if ss.get("memory_autosave", True):
            try:
                _write_context_memory(
                    {
                        "version": 1,
                        "saved_at": int(time.time()),
                        "summary": ss.get("memory_summary", "") or "",
                        "items": ss.get("memory_items", []) or [],
                    }
                )
            except Exception:
                pass

    st.subheader("Per-Question Scores")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Key Areas for Improvement")
    if all_weaknesses:
        top = Counter(all_weaknesses).most_common(6)
        st.write("\n".join([f"- {w} (x{c})" for w, c in top]))
    else:
        st.write("—")

    st.subheader("Detailed Feedback")
    transcripts: Dict[int, str] = ss.get("transcripts", {})
    submitted_audio: Dict[int, Any] = ss.get("submitted_audio", {})
    for i, q in enumerate(questions):
        f = feedback_map.get(i) or {}
        score = int(f.get("score", 0)) if f else 0
        label = (q[:90] + "…") if len(q) > 90 else q
        with st.expander(f"Question {i + 1}: {label}"):
            st.write(q)
            audio_payload = submitted_audio.get(i)
            audio_bytes = _audio_payload_bytes(audio_payload)
            if audio_bytes:
                st.caption("Recorded answer")
                mime = _audio_payload_mime(audio_payload) or None
                st.audio(audio_bytes, format=mime)
            st.markdown("**Transcript**")
            st.write(transcripts.get(i, "") or "—")
            st.markdown("**Score**")
            st.write(f"{score}/10" if score else "—")
            st.markdown("**Strengths**")
            strengths = f.get("strengths") or []
            st.write("- " + "\n- ".join(strengths) if strengths else "—")
            st.markdown("**Weaknesses**")
            weaknesses = f.get("weaknesses") or []
            st.write("- " + "\n- ".join(weaknesses) if weaknesses else "—")
            st.markdown("**Refined Sample Answer**")
            st.markdown(
                _render_refined_answer_html(
                    transcripts.get(i, "") or "",
                    f.get("refined_answer") or "",
                ),
                unsafe_allow_html=True,
            )
            perfect_answer = str(f.get("perfect_answer") or "").strip()
            pa_time_limit = int(f.get("time_limit_s") or TIME_LIMIT_OPTIONS.get(ss.get("time_limit_label", "60s"), 60))
            pa_range = f.get("perfect_answer_word_range") or []
            if isinstance(pa_range, list) and len(pa_range) >= 2:
                pa_min_words = int(pa_range[0])
                pa_max_words = int(pa_range[1])
            else:
                pa_min_words, pa_max_words, _ = _target_word_range_for_time_limit(pa_time_limit)
            st.markdown(f"**Perfect Answer (Resume + JD, ~{pa_time_limit}s)**")
            st.caption(f"Target length: {pa_min_words}-{pa_max_words} words | Current: {_word_count(perfect_answer)} words")
            st.markdown(_render_perfect_answer_html(perfect_answer), unsafe_allow_html=True)
            st.markdown("**Response Techniques**")
            techniques = f.get("techniques") or []
            st.write("- " + "\n- ".join(techniques) if techniques else "—")

    st.divider()
    if st.button("Start New Section", type="primary"):
        _reset_section_state()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _hide_streamlit_header_actions()
    st.title(APP_TITLE)

    _init_session_state()
    client, resume_text, jd_text = _render_sidebar()

    ss = st.session_state
    status = ss.get("status", STATUS_SETUP)

    if status == STATUS_SETUP:
        st.write(
            "Upload your resume and paste a job description in the sidebar, then click **Start Interview Section**.\n\n"
            "You can run without an API key in **Mock Mode** to validate the layout and flow."
        )
        with st.expander("Preview extracted resume text"):
            st.write((resume_text or ss.get("resume_text", ""))[:4000] or "—")
        with st.expander("Preview job description"):
            st.write((jd_text or ss.get("jd_text", ""))[:4000] or "—")
        return

    if status == STATUS_INTERVIEWING:
        _render_interview(client)
        return

    if status == STATUS_REVIEW:
        _render_review(client)
        return

    st.error(f"Unknown app status: {status}")


if __name__ == "__main__":
    main()
