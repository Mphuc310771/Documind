from pydantic import BaseModel, field_validator
from urllib.parse import urlparse

class QueryRequest(BaseModel):
    query: str
    provider: str = "auto"
    notebook_id: str = "default"
    search_web: bool = False

class QueryResponse(BaseModel):
    answer: str

class SynthesizeNotesRequest(BaseModel):
    notes: list[str]
    action: str
    provider: str = "auto"

class StudioGenerateRequest(BaseModel):
    content_type: str
    provider: str = "auto"
    notebook_id: str = "default"

class PodcastGenerateRequest(BaseModel):
    provider: str = "auto"
    notebook_id: str = "default"
    custom_instructions: str = ""

class SlideGenerateRequest(BaseModel):
    provider: str = "auto"
    notebook_id: str = "default"
    num_slides: int = 10
    chat_context: str | None = None


class NotebookCreateRequest(BaseModel):
    name: str
    id: str | None = None


class ChatMessageRequest(BaseModel):
    notebook_id: str = "default"
    role: str
    content: str = ""
    citations: list | None = None


class CanvasEditRequest(BaseModel):
    content: str = ""
    selection: str = ""
    instruction: str
    provider: str = "auto"


class TtsSpeakRequest(BaseModel):
    text: str
    voice: str = "male"  # male | female


class AuthRequest(BaseModel):
    username: str
    password: str


class QuickSummarizeRequest(BaseModel):
    text: str
    provider: str = "auto"


class UploadURLRequest(BaseModel):
    url: str
    notebook_id: str = "default"

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("URL không được để trống.")
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("URL phải bắt đầu bằng http:// hoặc https://")
        return value


class AppGenerateRequest(BaseModel):
    notebook_id: str = "default"
    app_prompt: str
    provider: str = "auto"


