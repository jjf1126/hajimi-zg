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

        data = self.filter_data_by_whitelist(request, whitelist)

        # --- 核心修复开始 ---
        
        # 1. 安全初始化 tools (修复 crash 问题)
        # 这里的关键是：如果 request.tools 为 None，whitelist 过滤后 data["tools"] 会是 None
        # 直接调用 setdefault...append 会报错，所以必须先确保它是个 list
        if data.get("tools") is None:
            data["tools"] = []

        # 2. 联网搜索模式处理
        if settings.search["search_mode"] and data.get("model", "").endswith("-search"):
            log(
                "INFO",
                "开启联网搜索模式 (OpenAI Endpoint)",
                extra={"key": self.api_key[:8], "model": data.get("model")},
            )
            
            # 检查是否已包含搜索工具，防止重复添加
            tools_list = data["tools"]
            has_search_tool = False
            for tool in tools_list:
                if isinstance(tool, dict) and ("googleSearch" in tool or "google_search" in tool):
                    has_search_tool = True
                    break
            
            if not has_search_tool:
                # 使用 googleSearch (驼峰) 适配 Gemini API
                tools_list.append({"googleSearch": {}})
            
            # 移除后缀
            data["model"] = data["model"].removesuffix("-search")

        # 3. 兜底逻辑：即使没开启搜索模式，只要检测到后缀也必须移除，否则 API 会报 404
        if data.get("model", "").endswith("-search"):
             data["model"] = data["model"].removesuffix("-search")
             
        # --- 核心修复结束 ---

        extra_log = {
            "key": self.api_key[:8],
            "request_type": "stream",
            "model": data.get("model"),
        }
        log("INFO", "流式请求开始", extra=extra_log)

        # 注意：这里使用的是 Google 的 OpenAI 兼容接口
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
