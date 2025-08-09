from pydantic import BaseModel, Field, HttpUrl
from typing import Literal, Optional, Dict
import uuid, time

JobType = Literal["separate", "enhance", "classify"]

class SeparationParams(BaseModel):
    model: Literal["nemo", "htdemucs"] = "nemo"
    max_stems: int = 5
    diarize: bool = False
    boost_speech: bool = True

class SoundMindJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_ts: float = Field(default_factory=lambda: time.time())
    job_type: JobType = "separate"
    # accept local file path or http(s) URL
    audio_path: str
    task: str = ""          # free‑text instruction (e.g., “isolate adult male + child laugh”)
    params: Dict = Field(default_factory=lambda: SeparationParams().model_dump())
    retry_count: int = 0
