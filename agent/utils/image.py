"""Image loading utilities."""

import os
from io import BytesIO
from typing import Optional

import PIL.Image
import requests


def load_image(input_data) -> Optional[PIL.Image.Image]:
    """Load an image from a PIL Image object, a URL, or a local file path.

    Args:
        input_data: A ``PIL.Image.Image``, an HTTP/HTTPS URL string, or a
                    local file-path string.

    Returns:
        A ``PIL.Image.Image`` in RGB mode, or ``None`` if loading failed or
        ``input_data`` is ``None`` / unrecognised.
    """
    if input_data is None:
        return None

    try:
        if isinstance(input_data, PIL.Image.Image):
            return input_data

        if isinstance(input_data, str) and input_data.startswith(("http://", "https://")):
            response = requests.get(input_data, timeout=10)
            response.raise_for_status()
            return PIL.Image.open(BytesIO(response.content)).convert("RGB")

        if isinstance(input_data, str) and os.path.exists(input_data):
            return PIL.Image.open(input_data).convert("RGB")

        return None

    except Exception as exc:
        print(f"Image load error: {exc}")
        return None
