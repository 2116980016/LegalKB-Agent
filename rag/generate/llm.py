"""LLM 适配层。

- EchoLLM：离线确定性实现，从 prompt 里抽取片段并拼出带引用的回答。
  存在的意义：没有 API Key 也能跑通整条链路 + 跑 CI + 让别人 clone 下来就能用。
- OpenAICompatLLM：任何 OpenAI 兼容端点（DashScope / DeepSeek / 智谱 / vLLM）。
  用标准库 urllib 实现，不引入额外 HTTP 依赖。

两者接口一致，`LLM_STREAM=1` 时都支持流式。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator, List, Optional

from rag.generate.prompt import REFUSAL

_SNIPPET = re.compile(r"\[片段(\d+)\]\s*([^\n]{0,160})")


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class EchoLLM:
    name = "echo"

    def complete(self, messages: List[dict]) -> LLMResponse:
        user = messages[-1]["content"] if messages else ""
        matches = _SNIPPET.findall(user)
        if not matches:
            return LLMResponse(text=REFUSAL)
        first_idx, first_text = matches[0]
        answer = f"根据参考资料，{first_text.strip()} [片段{first_idx}]"
        if len(matches) > 1 and len({m[0] for m in matches}) > 1:
            second_idx, second_text = matches[1]
            answer += f"\n\n补充依据：{second_text.strip()[:100]} [片段{second_idx}]"
        answer += "\n\n提示：以上为检索结果的归纳，具体适用请结合案件事实与最新法规确认。"
        return LLMResponse(text=answer, prompt_tokens=len(user), completion_tokens=len(answer))

    def stream(self, messages: List[dict]) -> Iterator[str]:
        text = self.complete(messages).text
        for piece in _chunks(text, 12):
            yield piece


class OpenAICompatLLM:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-plus",
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("缺少 API Key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = model

    def _request(self, messages: List[dict], stream: bool) -> urllib.request.Request:
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": self.temperature, "stream": stream},
            ensure_ascii=False,
        ).encode("utf-8")
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream" if stream else "application/json",
            },
            method="POST",
        )

    def _open(self, messages: List[dict], stream: bool):
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return urllib.request.urlopen(self._request(messages, stream), timeout=self.timeout)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                raise
            except urllib.error.URLError as e:  # 网络抖动
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 请求失败：{last_err}")

    def complete(self, messages: List[dict]) -> LLMResponse:
        with self._open(messages, stream=False) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def stream(self, messages: List[dict]) -> Iterator[str]:
        with self._open(messages, stream=True) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except Exception:
                    continue
                piece = delta.get("content")
                if piece:
                    yield piece


def _chunks(text: str, size: int) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


def build_llm(provider: str = "echo", **kwargs):
    provider = (provider or "echo").lower()
    if provider in {"echo", "mock", "offline"}:
        return EchoLLM()
    if provider in {"openai", "dashscope", "qwen", "deepseek", "zhipu"}:
        return OpenAICompatLLM(**kwargs)
    raise ValueError(f"未知的 llm provider: {provider}")
