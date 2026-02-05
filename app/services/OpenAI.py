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
        # 兼容 Pydantic 模型和字典
        if hasattr(data, "model_dump"):
            data_dict = data.model_dump()
        elif hasattr(data, "dict"):
            data_dict = data.dict()
        elif isinstance(data, dict):
            data_dict = data
        else:
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

        # 修复：使用 self 调用方法
        data = self.filter_data_by_whitelist(request, whitelist)

        # 修复：data 是字典，必须用 ["key"] 访问
        if settings.search["search_mode"] and data.get("model", "").endswith("-search"):
            log(
                "INFO",
                "开启联网搜索模式 (OpenAI Endpoint)",
                extra={"key": self.api_key[:8], "model": data["model"]},
            )
            # 修复：使用 googleSearch (驼峰)
            data.setdefault("tools", []).append({"googleSearch": {}})
            # 移除后缀
            data["model"] = data["model"].removesuffix("-search")

        # 确保移除后缀 (即使没开启搜索模式，只要带了后缀也要移除，防止报错)
        if data.get("model", "").endswith("-search"):
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
                try:
                    # 检查 HTTP 状态
                    if response.status_code != 200:
                         await response.aread()
                         response.raise_for_status()
                    
                    buffer = b""
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: ") :].strip()

                        if line == "[DONE]":
                            break

                        buffer += line.encode("utf-8")
                        try:
                            chunk_data = json.loads(buffer.decode("utf-8"))
                            buffer = b""
                            yield chunk_data

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            log("ERROR", "流式处理错误", extra={"error": str(e)})
                            raise e
                except Exception as e:
                    if not response.is_closed:
                        await response.aread()
                    raise e
                finally:
                    log("info", "流式请求结束")
