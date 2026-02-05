import json
import os
import time  # 新增导入
from app.models.schemas import ChatCompletionRequest
from dataclasses import dataclass
from typing import Optional
import httpx
import secrets
import string
import app.config.settings as settings

# 新增导入：为了在搜索模式下使用原生 API
from app.services.gemini import GeminiClient

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
        # ----------------------------------------------------------------------
        # 1. 拦截搜索请求：转交 Gemini 原生客户端处理
        # ----------------------------------------------------------------------
        # 原因：Google 的 OpenAI 兼容接口 (/openai/chat/completions) 不支持 googleSearch 工具。
        # 只有原生接口 (generateContent) 支持。因此需要在此处做协议转换。
        if settings.search["search_mode"] and request.model.endswith("-search"):
            log("INFO", "检测到搜索请求，切换至 Gemini 原生协议", extra={"model": request.model})
            
            try:
                # 初始化原生客户端
                gemini_client = GeminiClient(self.api_key)
                
                # 转换消息格式 (OpenAI Messages -> Gemini Contents)
                contents, system_instruction = gemini_client.convert_messages(
                    request.messages, 
                    use_system_prompt=True,
                    model=request.model
                )
                
                # 获取安全设置 (尝试从 settings 获取，如果没有则为 None)
                safety_settings = getattr(settings, "SAFETY_SETTINGS", None)
                
                # 调用 Gemini 原生流式接口
                # GeminiClient 内部会自动处理 googleSearch 工具注入和 model 后缀移除
                async for response_wrapper in gemini_client.stream_chat(
                    request, contents, safety_settings, system_instruction
                ):
                    # --- 响应适配器 (Gemini -> OpenAI Chunk) ---
                    text = response_wrapper.text
                    finish_reason = response_wrapper.finish_reason
                    
                    # 映射 finish_reason
                    openai_finish_reason = None
                    if finish_reason == "STOP":
                        openai_finish_reason = "stop"
                    elif finish_reason == "MAX_TOKENS":
                        openai_finish_reason = "length"
                    elif finish_reason: # 其他情况
                         openai_finish_reason = finish_reason.lower()

                    # 仅当有内容或结束时生成 chunk
                    if text or openai_finish_reason:
                        chunk = {
                            "id": f"chatcmpl-{generate_secure_random_string(29)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text} if text else {},
                                    "finish_reason": openai_finish_reason
                                }
                            ]
                        }
                        yield chunk
                return  # 搜索请求处理完毕，直接返回

            except Exception as e:
                log("ERROR", "Gemini 原生协议转发失败", extra={"error": str(e)})
                raise e

        # ----------------------------------------------------------------------
        # 2. 普通请求：继续使用 OpenAI 兼容接口
        # ----------------------------------------------------------------------
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

        # 即使不开启搜索模式，如果模型名带 -search，也必须移除后缀，否则 API 报错
        if data.get("model", "").endswith("-search"):
             data["model"] = data["model"].removesuffix("-search")

        extra_log = {
            "key": self.api_key[:8],
            "request_type": "stream",
            "model": data.get("model"),
        }
        log("INFO", "流式请求开始 (OpenAI Endpoint)", extra=extra_log)

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
