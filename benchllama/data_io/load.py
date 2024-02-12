import logging

from typing import List
from datasets import load_dataset

_ALLOWED_LANGUAGES = ["python", "js", "java", "go", "cpp", "rust"]
log = logging.getLogger(__name__)

class Loader():
    dataset = None

    def __init__(self, name="bigcode/humanevalpack", language="python"):
        if language not in _ALLOWED_LANGUAGES:
            raise ValueError("Language {} is not supported".format(language))
        self.dataset = load_dataset(name, language)["test"]
        log.info("Loading dataset {} for language {}".format(name, language))

    def get_data(self):
        return self.dataset