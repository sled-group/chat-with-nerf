import json


def fix_brackets(json_str):
    open_brackets = json_str.count("{")
    close_brackets = json_str.count("}")

    if open_brackets > close_brackets:
        json_str += "}" * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        json_str = json_str.rstrip("}")
        json_str = json_str[: json_str.rfind("{")] + json_str[json_str.rfind("}") + 1 :]

    return json_str


def robust_json_loads(json_str):
    json_str = fix_brackets(json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Invalid JSON")
        return None
