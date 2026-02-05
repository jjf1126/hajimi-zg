import json
import os
from app.models.schemas import ChatCompletionRequest
from dataclasses import dataclass
from typing import Optional
import httpx
import secrets
import string
import app.config.settings as settings

from app.utils.logging import log


def generate_secure_random_string(length):
    all_characters = string.ascii_letters + string.digits
    secure_random_string = "".join(
        secrets.choice(all_characters) for _ in range(length)
    )
    return secure_random_string


@dataclass
class GeneratedText:
    text: str
    finish_reason: Optional[str] = None


class OpenAIClient:
    AVAILABLE_MODELS = []
    EXTRA_MODELS = os.environ.get("EXTRA_MODELS", "").split(",")

    def __init__(self, api_key: str):
        self.api_key = api_key

    def filter_data_by_whitelist(self, data, allowed_keys):
        """
        根据白名单过滤字典。
        Args:
            data (object or dict): 原始数据（可能是 Pydantic 模型或字典）。
            allowed_keys (list or set): 包含允许保留的键名的列表或集合。
        Returns:
            dict: 只包含白名单中键的新字典。
        """
        # 如果是 Pydantic 模型，先转为字典
        if hasattr(data, "model_dump"):
            data_dict = data.model_dump()
        elif hasattr(data, "dict"):
            data_dict = data.dict()
        elif isinstance(data, dict):
            data_dict = data
        else:
            # 尝试通过 __dict__ 获取或报错，视具体实现而定，这里假设它是类对象
            data_dict = getattr(data, "__dict__", {})

        allowed_keys_set = set(allowed_keys)
        filtered_data = {
            key: value for key, value in data_dict.items() if key in allowed_keys_set
        }
        return filtered_data

    # 真流式处理
    async def stream_chat(self, request: ChatCompletionRequest):
        whitelist = [
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "stream",
            "tools",
            "reasoning_effort",
            "top_k",
            "presence_penalty",
        ]

        # 1. 修复：调用 filter 时传入 self (虽然之前定义在类里没加 self 但没加 @staticmethod，这里作为实例方法调用)
        # 注意：原代码定义 filter_data_by_whitelist 没有 self，但在类里。如果作为实例方法调用会报错。
        # 我在上面给它加了 self。
        data = self.filter_data_by_whitelist(request, whitelist)

        # 2. 修复：data 是字典，必须用 ["key"] 访问，不能用 .key
        # 3. 修复：模型后缀检测逻辑
        if settings.search["search_mode"] and data.get("model", "").endswith("-search"):
            log(
                "INFO",
                "开启联网搜索模式 (OpenAI Endpoint)",
                extra={"key": self.api_key[:8], "model": data["model"]},
            )
            # 4. 修复：使用 googleSearch (驼峰) 适配新版 API
            data.setdefault("tools", []).append({"googleSearch": {}})
            
            # 移除后缀
            data["model"] = data["model"].removesuffix("-search")

        # 真流式请求处理逻辑
        extra_log = {
            "key": self.api_key[:8],
            "request_type": "stream",
            "model": data.get("model"),
        }
        log("INFO", "流式请求开始", extra=extra_log)

        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", url, headers=headers, json=data, timeout=600
            ) as response:
                buffer = b""  # 用于累积可能不完整的 JSON 数据
                try:
                    # 检查 HTTP 状态码，如果是 4xx/5xx 直接报错
                    if response.status_code != 200:
                         await response.aread()
                         response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip():  # 跳过空行 (SSE 消息分隔符)
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: ") :].strip()  # 去除 "data: " 前缀

                        # 检查是否是结束标志，如果是，结束循环
                        if line == "[DONE]":
                            break

                        buffer += line.encode("utf-8")
                        try:
                            # 尝试解析整个缓冲区
                            chunk_data = json.loads(buffer.decode("utf-8"))
                            # 解析成功，清空缓冲区
                            buffer = b""

                            yield chunk_data

                        except json.JSONDecodeError:
                            # JSON 不完整，继续累积到 buffer
                            continue
                        except Exception as e:
                            log(
                                "ERROR",
                                "流式处理期间发生错误",
                                extra={
                                    "key": self.api_key[:8],
                                    "request_type": "stream",
                                    "model": data.get("model"),
                                    "error": str(e)
                                },
                            )
                            raise e
                except Exception as e:
                    raise e
                finally:
                    log("info", "流式请求结束")
