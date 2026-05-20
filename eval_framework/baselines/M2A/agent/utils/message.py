import base64
import re

def extract_image_id(text: str) -> int:
    """
    Extract image id from string of format <image{id}>.
    
    Example: 
        extract_image_id("<image123>") -> 123
    """
    pattern = r'<image(\d+)>'
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Wrong format.")


def encode_image_to_base64(image_path: str) -> str:
    """Read image file and encode to base64 http body"""
    import mimetypes
    from pathlib import Path

    path = Path(image_path)
    with open(path, 'rb') as f:
        image_data = f.read()

    # Determine MIME type
    mime_type = mimetypes.guess_type(str(path))[0]
    if mime_type is None:
        # Default to jpeg if can't determine
        mime_type = 'image/jpeg'
    elif mime_type.startswith('text/'):
        mime_type = 'image/jpeg'

    return f"data:{mime_type};base64,{base64.b64encode(image_data).decode('utf-8')}"

