from sys import version_info

import litellm

CHARACTERS_PER_TOKEN_ASSUMPTION: float = 4.0
EXTRA_TOKENS_FROM_USER_ROLE: int = 7

MODEL_COST_MAP = litellm.get_model_cost_map("")

DEFAULT_VERTEX_SAFETY_SETTINGS: list[dict[str, str]] = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
]

IS_PYTHON_BELOW_312 = version_info < (3, 12)
