"""Monkeypatch to ensure multimodal message placeholders exist when images are present.

Some datasets include images but the conversation messages lack the expected image placeholder
parts (e.g., {"type": "image"}). This causes a mismatch between image features and image
placeholder tokens at tokenization time. This module patches `trl.data_utils.prepare_multimodal_messages`
to guarantee at least one image placeholder is inserted when images are present.
"""
from __future__ import annotations

def patch() -> None:
    try:
        from trl import data_utils
    except Exception:
        return

    if getattr(data_utils.prepare_multimodal_messages, "_patched_with_placeholder_fix", False):
        return

    _orig = data_utils.prepare_multimodal_messages

    def _wrapped(messages, num_images):
        _orig(messages, num_images)
        try:
            if not num_images:
                return
            # Check if any part already contains an image placeholder
            found = False
            for msg in messages:
                cont = msg.get("content")
                if isinstance(cont, list):
                    for part in cont:
                        if isinstance(part, dict) and part.get("type") == "image":
                            found = True
                            break
                if found:
                    break
            if found:
                return

            # Insert placeholders into the first user message (or first message as fallback)
            target = None
            for msg in messages:
                if msg.get("role") == "user":
                    target = msg
                    break
            if target is None and messages:
                target = messages[0]

            if target is None:
                return

            content = target.get("content")
            placeholders = [{"type": "image"}] * int(num_images)
            if isinstance(content, str):
                target["content"] = [*placeholders, {"type": "text", "text": content}]
            elif isinstance(content, list):
                target["content"] = [*placeholders, *content]
            else:
                # unknown content format; set a new content with image placeholder and an empty text
                target["content"] = [*placeholders, {"type": "text", "text": ""}]
        except Exception:
            # Don't fail the collator if our patch has issues; fall back to original behavior
            return

    _wrapped._patched_with_placeholder_fix = True
    data_utils.prepare_multimodal_messages = _wrapped


if __name__ == "__main__":
    patch()
