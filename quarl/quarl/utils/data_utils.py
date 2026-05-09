import copy
import json
import re


def detect_language(text):
    # 用字符统计
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    english_chars = re.findall(r"[a-zA-Z]", text)
    if len(chinese_chars) > len(english_chars) // 5:
        return "chinese"
    elif len(english_chars) > len(chinese_chars):
        return "english"

    return "unknown"


def remove_last_incomplete_sentence(text):
    delimiters = [
        ".",
        "!",
        "?",
        "。",
        "！",
        "？",
        "…",
    ]
    idx = len(text) - 1
    # 从后向前查找第一个标点
    while idx >= 0 and text[idx] not in delimiters:
        idx -= 1
    return text[: idx + 1] if idx >= 0 else text


def remove_prefix(content, prefix):
    for i in reversed(range(min(len(content), len(prefix)))):
        if content[: i + 1] == prefix[-(i + 1) :]:
            content = content[i + 1 :]
            break
    return content


def extract_thought_and_answer(text):
    # 定义正则模式
    pattern = r"<think>\n(.*?)\n</think>(.*)"

    if text.count("<think>") == 1 and text.count("</think>") == 1:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thought_content = match.group(1)
            answer_content = match.group(2)
            return [thought_content, answer_content]
        else:
            return None
    else:
        return None


def detect_image_search(text):
    """
    参数:
        text (str): 需要处理的源字符串

    返回:
        bool: 是否包含图搜内容
    """
    # 定义需要检查的模式
    patterns = [
        # 匹配 {{Agent_ImageSearch数字(任意内容)}}
        r"{{Agent_ImageSearch\d+\([^)]+\)}}",
        r"{{Agent_End}}",  # 匹配 {{Agent_End}}
        r"{{Agent_ImageInsert}}",  # 匹配 {{Agent_ImageInsert}}
    ]

    # 依次检查每个模式
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    if result != text:  # 如果结果和原始字符串不同，则包含图搜内容
        return 1
    else:
        return 0


def rm_image_search_content(text):
    """
    参数:
        text (str): 需要处理的源字符串

    返回:
    """
    # 定义需要检查的模式
    patterns = [
        # 匹配 {{Agent_ImageSearch数字(任意内容)}}
        r"{{Agent_ImageSearch\d+\([^)]+\)}}",
        r"{{Agent_End}}",  # 匹配 {{Agent_End}}
        r"{{Agent_ImageInsert}}",  # 匹配 {{Agent_ImageInsert}}
    ]

    # 依次检查每个模式
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    return result


def img_search_format_error(response_txt):
    s = response_txt
    # Image_Search1 pair cnt
    img_search1_cnt = len(re.findall("\\{\\{Agent_ImageSearch1\\(.*?\\)\\}\\}", s))
    img_search2_cnt = len(re.findall("\\{\\{Agent_ImageSearch2\\(.*?\\)\\}\\}", s))
    img_search1_pair_cnt = len(re.findall("\\{\\{Agent_ImageSearch1\\(.*?\\)\\}\\}[\S\s]*?{{Agent_End}}", s))
    img_insert_cnt = len(re.findall("\\{\\{Agent_ImageInsert\\}\\}", s))
    img_end_cnt = len(re.findall("\\{\\{Agent_End\\}\\}", s))

    if img_search1_cnt != img_search1_pair_cnt:
        return 0

    if img_search1_cnt > 0 and img_insert_cnt == 0:
        return 0

    if img_search1_cnt == 0 and (img_end_cnt > 0 or img_insert_cnt > 0):
        return 0

    return 1

def extract_docs_and_query(prompt_str: str, raw_prompt: list = []):
    doc = re.search(r'Passages:\n(.*?)\nQuestion:', prompt_str, re.DOTALL)
    query = re.search(r'Question:\s*(.*?)\n', prompt_str, re.DOTALL)
    
    try:
        doc = doc.group(1).strip()
    except:
        doc = ""

    try:
        query = query.group(1).strip()
    except:
        query = ""
        
    if query.endswith("assistant"):
        query = query[:-9].strip()

    return doc, query