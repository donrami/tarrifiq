from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    a: str = ""
    chroma_server_nofile: Optional[int] = None

    @validator("chroma_server_nofile", pre=True)
    def empty_str_to_none(cls, v: str) -> Optional[str]:
        if type(v) is str and v.strip() == "":
            return None
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

print(Settings())
