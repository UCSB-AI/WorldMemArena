from datetime import datetime
import json
import os
import re
from typing import Optional, Literal
from ..config import TIME_FMT
from ..utils.message import extract_image_id
from ..embeddings.multimodal import encode_image_to_base64


class ImageManager:
    def __init__(self):
        self.image_paths = []
        self.image_path_to_id = {}

    def save(self, path: str):
        data = {
            'image_paths': self.image_paths,
            'image_path_to_id': self.image_path_to_id,
        }

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Save to {path}")

    def load(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.image_paths = data.get('image_paths', [])
            self.image_path_to_id = data.get('image_path_to_id', {})

            print(f"Load from {path}")
            return True

        except FileNotFoundError:
            print(f"Not exists: {path}")
            return False

    def refresh(self):
        self.__init__()

    def image_token_to_image(self, image_token: Optional[str]) -> str|None:
        if image_token is None:
            return None
        try:
            id = extract_image_id(image_token)
        except Exception:
            raise Exception("Invalid image token: Wrong format!")

        try:
            return self.image_paths[id]
        except Exception:
            raise Exception("Invalid image token: No specified image!")

    def _image_to_image_token(self, image: str) -> str:
        try:
            return f"<image{self.image_path_to_id[image]}>"
        except:
            raise f"Unregistered image: {image}"

    def _register_image_if_not(self, image_path_or_url: str) -> int:
        """
        Register a new image if not registered.

        Returns:
            Image id, whether newly registered or not.
        """
        if image_path_or_url not in self.image_path_to_id:
            self.image_paths.append(image_path_or_url)
            self.image_path_to_id[image_path_or_url] = len(self.image_paths)-1
        return self.image_path_to_id[image_path_or_url]

    def image_to_image_token(self, image: str) -> str:
        """
        Get image token, automatically register if not yet.
        """
        self._register_image_if_not(image)
        return self._image_to_image_token(image)

    def format_to_msg_content(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        in_content: bool = True,
        **additional_fields
    ) -> list[dict]:
        """
        image: image url or path
        """
        content = []
        json_msg = {}

        if text:
            json_msg["text"] = text
        if image:
            json_msg["image_token"] = self.image_to_image_token(image)
            json_msg["image"] = "<image>"

        if additional_fields:
            if in_content:
                additional_fields["content"] = json_msg
                json_msg = additional_fields
            else:
                for k, v in additional_fields.items():
                    if isinstance(v, datetime):
                        v = v.strftime(TIME_FMT)
                    json_msg[k] = v

        msg = json.dumps(json_msg)
        chunks = re.split('("<image>")', msg)

        for chunk in chunks:
            if chunk == '"<image>"':
                content.append(
                    {"type": "image", "url": image}
                )
            else:
                content.append(
                    {"type": "text", "text": chunk}
                )
        return content

    def format_image(self, img_path_or_url: Optional[str] = None) -> list[dict]:
        """
        image: image url or path
        """
        if img_path_or_url is None:
            return []

        if not img_path_or_url.startswith("http"):
            img_path_or_url = encode_image_to_base64(img_path_or_url)

        return self.format_to_msg_content(image=img_path_or_url, in_content=False)
        

    def format_obj_to_content(self, x, images: list[str]) -> list[dict]:
        """
        replace str("<image>") with image content format
        images: list of image url/path
        """
        content = []

        msg = json.dumps(x, indent=2)
        chunks = re.split('("<image>")', msg)

        assert len(chunks) == 2*len(images) + 1

        cur_images = (image for image in images)

        for chunk in chunks:
            if chunk == '"<image>"':
                content += self.format_image(next(cur_images))
            else:
                content.append(
                    {"type": "text", "text": chunk}
                )
        return content

    def format_msg_to_content(self, msg: str) -> str | list[dict]:
        """
        msg: str containing image token
        """
        chunks = re.split(r'(<image\d+>)', msg)
        chunks = [chunk for chunk in chunks if chunk]

        content = []
        for chunk in chunks:
            if re.match(r'^<image\d+>$', chunk):
                content.append({"type": "text", "text": f"{chunk}: "})
                content += self.format_image(
                    self.image_token_to_image(chunk)
                )
            else:
                content.append(
                    {"type": "text", "text": chunk}
                )
        return content
