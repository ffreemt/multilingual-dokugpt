"""Load sk-/pk- key."""
# pylint: disable=invalid-name
from os import getenv
from typing import Optional

from dotenv import load_dotenv

sk_base = "https://api.openai.com/v1"
pk_base = "https://api.pawan.krd/v1"


def load_api_key(env_var: Optional[str] = None):
    """Load OPENAI_API_KEY/SK-/PK- key.

    if env_var is None, load from .env
        order: "OPENAI_API_KEY", SK_KEY, PK_KEY
    else:
        dotenv_values("env_var") | os.getenv("env_var")
    """
    # with override=True .env has higher priority
    # than os.get(...)
    load_dotenv(override=True)

    if env_var is not None:
        return getenv(str(env_var))

    _ = [
        "OPENAI_API_KEY",
        "SK_KEY",
        "PK_KEY",
    ]

    api_key = None
    for api_key in map(getenv, _):
        if api_key:
            break

    return api_key
