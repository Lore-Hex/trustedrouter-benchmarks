"""Convert BFCL function schemas to OpenAI `tools` format — vendored from BFCL's
own `convert_to_tool` / `_cast_to_openai_type` (model_handler/utils.py, Apache-2.0)
so the model sees exactly the schema the leaderboard sends.

BFCL's schema uses its own type names ("dict", "float", "tuple", "any", …) and
allows "." in function names; OpenAI's tools API needs JSON-Schema types and
forbids ".". This applies BFCL's exact transform: name "."→"_", parameters.type
→ "object", and `GORILLA_TO_OPENAPI` type mapping (recursively). The ORIGINAL
(untransformed) schema is what the AST checker scores against — only the model
sees the converted one.
"""
from __future__ import annotations

import copy
import re

# Vendored from bfcl_eval/constants/type_mappings.py (GORILLA_TO_OPENAPI).
GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number",
    "string": "string", "boolean": "boolean", "bool": "boolean",
    "array": "array", "list": "array", "dict": "object", "object": "object",
    "tuple": "array", "any": "string", "byte": "integer", "short": "integer",
    "long": "integer", "double": "number", "char": "string",
    "ArrayList": "array", "Array": "array", "HashMap": "object",
}


def _cast_to_openai_type(properties: dict, mapping: dict) -> dict:
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]
            if var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] = (
                    properties[key].get("description", "") + " This is a float type value."
                )
            properties[key]["type"] = mapping.get(var_type, "string")

        if properties[key]["type"] in ("array", "object"):
            if "properties" in properties[key]:
                properties[key]["properties"] = _cast_to_openai_type(properties[key]["properties"], mapping)
            elif "items" in properties[key]:
                properties[key]["items"]["type"] = mapping.get(properties[key]["items"]["type"], "string")
                if properties[key]["items"]["type"] == "array" and "items" in properties[key]["items"]:
                    properties[key]["items"]["items"]["type"] = mapping.get(
                        properties[key]["items"]["items"]["type"], "string"
                    )
                elif properties[key]["items"]["type"] == "object" and "properties" in properties[key]["items"]:
                    properties[key]["items"]["properties"] = _cast_to_openai_type(
                        properties[key]["items"]["properties"], mapping
                    )
    return properties


def sanitize_name(name: str) -> str:
    """OpenAI tools API forbids "." in names (BFCL's checker re-applies the same
    rewrite to the ground-truth name, so this stays consistent)."""
    return re.sub(r"\.", "_", name) if "." in name else name


def to_openai_tools(functions: list[dict]) -> list[dict]:
    functions = copy.deepcopy(functions)
    tools = []
    for item in functions:
        params = item.get("parameters", {"type": "object", "properties": {}})
        params["type"] = "object"
        params["properties"] = _cast_to_openai_type(params.get("properties", {}), GORILLA_TO_OPENAPI)
        tools.append({
            "type": "function",
            "function": {
                "name": sanitize_name(item["name"]),
                "description": item.get("description", ""),
                "parameters": params,
            },
        })
    return tools
