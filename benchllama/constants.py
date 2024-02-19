from enum import Enum

PROMPT_EVAL_DURATION = "prompt_eval_duration"
PROMPT_EVAL_COUNT = "prompt_eval_count"
EVAL_COUNT = "eval_count"
EVAL_DURATION = "eval_duration"
COMPLETION = "completion"
PROMPT_EVAL_RATE = "prompt_eval_rate"
EVAL_RATE = "eval_rate"

class Language(str, Enum):
    python = "python",
    js = "js",
    java = "java",
    go = "go",
    cpp = "cpp"


class Result(Enum):
    SUCCESS = 1
    FAILURE = 0