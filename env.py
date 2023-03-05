from pydantic import BaseSettings


class Env(BaseSettings):
    SECRET_KEY: str = '5@^uN`q(x&T[+clge(HID(QIv)sP9.SDFOk15k?"H!Z?|vr/hk6t?TUPdsd"@7T'
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    PEXELS_AUTH_KEY: str = "NCzuoY3ZRrUy2AD2FiSR27QNzc2OMYJJ7rGxi1qkYT36Q2EM8PQTG0BR"

