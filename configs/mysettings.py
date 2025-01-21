from pydantic_settings import BaseSettings
from pydantic import SecretStr

"""Classes below loads in environment variables of the same name
    and validates the data types upon loading
    """


class NeptuneSettings(BaseSettings):
    """Load Neptune AI settings from environment variables"""

    NEPTUNE_PROJECT: str = "Test Project"
    NEPTUNE_TOKEN: SecretStr


class HuggingFaceSettings(BaseSettings):
    """Loads Hugging Face settings from environment variables"""

    HF_TOKEN: SecretStr
