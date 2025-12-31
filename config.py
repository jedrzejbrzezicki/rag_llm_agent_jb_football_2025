from pydantic import BaseModel
from abc import ABC
import yaml

class ConfigModel(BaseModel, ABC):
    CHROMA_PATH: str
    FILE_PATHS: list[str]
    EMBEDDING_MODEL_NAME: str
    MODEL_NAME: str
    TOKENIZER_NAME: str

    class Config:
        arbitrary_types_allowed = True

class Config(ConfigModel):
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        super().__init__(
            CHROMA_PATH=config["CHROMA_PATH"],
            FILE_PATHS=config["FILE_PATHS"],
            EMBEDDING_MODEL_NAME=config["EMBEDDING_MODEL_NAME"],
            MODEL_NAME=config["MODEL_NAME"],
            TOKENIZER_NAME=config["TOKENIZER_NAME"]
        )