"""Prompt 构造。

三条约束缺一不可：
1. 只依据参考资料回答（防幻觉）；
2. 必须标注 [片段n]（可溯源）；
3. 资料不足要明确拒答，且给出固定句式（便于程序判定，而不是靠语义猜测）。
"""
from __future__ import annotations

SYSTEM_PROMPT = """你是一名严谨的法律咨询助手。你必须遵守以下规则：
1. 只能依据【参考资料】回答，不得使用资料之外的知识，不得推测。
2. 每个结论后面必须标注来源编号，格式为 [片段1]、[片段2]（可多个）。
3. 如果参考资料无法回答问题，只输出这一句：根据现有资料无法回答。
4. 不要编造法条编号；引用编号必须来自参考资料中真实存在的片段。
5. 回答用中文，先给结论，再给依据，最后给风险提示（如有时效性/地域性限制）。
6. 回答不超过 80 字，先给结论。"""


USER_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}

请严格依据参考资料作答，并在每条结论后标注 [片段n]。"""

REFUSAL = "根据现有资料无法回答"


def build_messages(question: str, context_block: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context_block, question=question)},
    ]
