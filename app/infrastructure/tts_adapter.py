import re
import asyncio
import logging
import edge_tts

logger = logging.getLogger(__name__)

VI_VOICES = {
    "male": "vi-VN-NamMinhNeural",
    "female": "vi-VN-HoaiMyNeural",
    "A": "vi-VN-HoaiMyNeural",
    "B": "vi-VN-NamMinhNeural",
}

MAX_TTS_CHARS = 800
HOST_TAG_RE = re.compile(
    r"^\[(?:Lan|Minh|Host\s*[AB]|MC\s*[AB]|A|B)\]\s*",
    re.IGNORECASE,
)


def clean_tts_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"[*#_`~]", " ", text)
    text = HOST_TAG_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS].rsplit(" ", 1)[0] + "..."
    return text


def host_to_voice_key(host: str) -> str:
    h = (host or "A").strip().upper()
    return "B" if h == "B" else "A"


async def synthesize_speech(text: str, voice_key: str = "male", retries: int = 3) -> bytes:
    """Synthesize Vietnamese speech via Microsoft Edge neural voices."""
    clean = clean_tts_text(text)
    if not clean:
        return b""

    if voice_key in VI_VOICES:
        voice = VI_VOICES[voice_key]
    else:
        voice = VI_VOICES["male" if voice_key == "male" else "female"]

    last_err = None
    for attempt in range(retries):
        audio = bytearray()
        try:
            communicate = edge_tts.Communicate(clean, voice)
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    audio.extend(chunk["data"])
            if audio:
                return bytes(audio)
            last_err = RuntimeError("No audio was received")
        except Exception as exc:
            last_err = exc
            logger.warning("TTS attempt %s/%s failed (%s): %s", attempt + 1, retries, voice, exc)
        if attempt + 1 < retries:
            await asyncio.sleep(1.2 * (attempt + 1))

    if last_err:
        raise last_err
    return b""


async def synthesize_to_file(text: str, filepath: str, voice_key: str = "A") -> bool:
    """Write MP3 bytes to filepath. Returns True on success."""
    data = await synthesize_speech(text, voice_key=voice_key)
    if not data:
        return False
    with open(filepath, "wb") as f:
        f.write(data)
    return True
