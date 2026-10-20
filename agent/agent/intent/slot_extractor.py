"""从用户这一句话里抽取槽位值。

多轮理赔每轮只填「第一个还空着的槽位」，因此抽取函数与槽位一一对应，
明天接入大模型后可以在这里加 LLM 抽取作为兜底。
"""

from __future__ import annotations

import re

CN_NUMBER = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

FULL_DAMAGE_KEYWORDS = ("全坏", "全烂", "都坏了", "都烂了", "全坏了")
NO_PHOTO_KEYWORDS = ("传不上", "上传不了", "无法上传", "没有照片", "没拍", "照片没", "不传了")
PHOTO_KEYWORDS = ("照片", "图片", "拍照", "上传了", "已上传")


def extract_order_id(text: str) -> str | None:
    match = re.search(r"(\d{10,})", text)
    return match.group(1) if match else None


def extract_damage_ratio(text: str) -> float | None:
    """把「一半」「三成」「30%」「全坏」统一转成 0~1 的小数。"""
    percent = re.search(r"(\d{1,3})\s*%", text)
    if percent:
        return min(1.0, max(0.0, int(percent.group(1)) / 100))

    if "一半" in text:
        return 0.5

    if any(keyword in text for keyword in FULL_DAMAGE_KEYWORDS):
        return 1.0

    cheng = re.search(r"([一二两三四五六七八九十])\s*成", text)
    if cheng:
        return CN_NUMBER[cheng.group(1)] / 10

    return None


def extract_evidence(text: str) -> str | None:
    if any(keyword in text for keyword in NO_PHOTO_KEYWORDS):
        return "文字描述（无照片）"
    if any(keyword in text for keyword in PHOTO_KEYWORDS):
        return "照片"

    stripped = text.strip()
    if len(stripped) >= 4:
        return f"文字描述：{stripped[:40]}"
    return None


def extract_value(slot: str, text: str, slots: dict | None = None) -> str | float | None:
    """按槽位名分派抽取逻辑，抽不到返回 None。"""
    if slot == "order_id":
        return extract_order_id(text)
    if slot == "damage_ratio":
        return extract_damage_ratio(text)
    if slot == "evidence":
        return extract_evidence(text)
    return None
