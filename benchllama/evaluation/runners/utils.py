import re


def is_instruct_model(row):
    return row["instruction"] == row["prompt"]


def parse_code(row):
    try:
        text = row["completion"]
        pattern = r"```([\w\s]*)\n([\s\S]+?)```"
        snippets = re.findall(pattern, text)
        return snippets[0][1]
    except Exception:
        return ""

def get_prompt_and_completion(row):
    if is_instruct_model(row):
        prompt = ""
        completion = parse_code(row)
    else:
        prompt = row["prompt"]
        completion = row["completion"]
    return prompt, completion