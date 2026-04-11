from pydantic import BaseSettings

class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()