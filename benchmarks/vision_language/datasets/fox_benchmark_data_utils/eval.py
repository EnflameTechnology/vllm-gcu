# from doctextVQAeval import VQAEval
import re
import logging

import nltk
import jieba

from nltk.metrics import precision, recall, f_measure
from nltk.translate import meteor_score


def contain_chinese_string(text):
    # 使用正则表达式匹配中文字符
    chinese_pattern = re.compile(r"[\u4e00-\u9fa5]")
    return bool(chinese_pattern.search(text))


inline_reg = re.compile(r"\\\((.*?)(?<!\\)\\\)")
display_reg = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
table_reg = re.compile(r"\\begin\{tabular\}(.+?)(?:\\end\{tabular\}|$)", re.S)


def split_text(pages, a_type):
    """
    Split a list of pages into text, inline math, display math, and table blocks.

    Args:
        pages: The pages to split.
    """
    text, math, table = [], [], []
    for page in pages:
        for i, reg in enumerate([inline_reg, display_reg, table_reg]):
            matches = "\n".join(reg.findall(page[a_type]))
            if i == 2:
                table.append(matches)
            elif i == 1:
                math[-1] += matches
            else:
                math.append(matches)
        page_str = page[a_type]
        text.append(page_str.strip())
    return text, math, table


def nougat_per_metrics(pred, gt, minlen=1, heavy_mode: int = 2):
    """
    Args:
    - heavy_mode:
        0 is clean mode, only similar, bleu, f1
        1 is normal, do not include edit_dist
        2 is heavy, total
    """
    metrics = {}

    if len(pred) < minlen or len(gt) < minlen:
        return metrics

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    if heavy_mode >= 1:
        # try:
        metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)
        # except LookupError:
        #     metrics["meteor"] = np.nan

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    if heavy_mode >= 1:
        metrics["precision"] = precision(reference, hypothesis)
        metrics["recall"] = recall(reference, hypothesis)
    if heavy_mode == 2:
        # 速度太慢
        metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(
            len(pred), len(gt)
        )
    return metrics


def doc_text_eval(results):
    """ """
    result = []
    for ann in results:
        try:
            ans = nougat_per_metrics(ann["predict"], ann["answer"])
            if len(ans) == 0:
                continue
            result.append(ans)
        except Exception as error:
            logging.exception("ERROR!!! Check yout output!!!")
            raise error

    mean_dict = {}
    mean_dict["eval question num"] = len(result)
    for k, v in result[0].items():
        mean_dict[k] = 0

    for each in result:
        for k, v in each.items():
            mean_dict[k] += v

    for k, v in mean_dict.items():
        if k == "eval question num":
            continue
        mean_dict[k] /= len(result)

    return mean_dict
