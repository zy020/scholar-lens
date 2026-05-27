from __future__ import annotations

import json
import re


def extract_json_from_llm_output(text: str) -> dict:
    """Extract JSON from LLM output, handling markdown code fences."""
    json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    json_str = json_match.group(1) if json_match else text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}
