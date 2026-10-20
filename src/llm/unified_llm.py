#!/usr/bin/env python3
"""
统一大模型调用类
支持本地llama.cpp、ollama和远程大模型API

用法:
    from unified_llm import LLM

    # 本地llama.cpp
    llm = LLM(backend="llama", base_url="http://localhost:8080")
    result = llm.generate("你好")

    # 本地ollama
    llm = LLM(backend="ollama", base_url="http://localhost:11434", model="qwen2.5:7b")
    result = llm.generate("介绍一下自己")

    # 远程OpenAI兼容API
    llm = LLM(backend="openai", base_url="https://api.openai.com/v1",
             model="gpt-4", api_key="your-key")
    result = llm.generate("分析这个数据")
"""
import argparse
import json
import time
from typing import Optional, List, Dict, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum
import urllib.request
import urllib.error


class LLMBackend(Enum):
    """支持的大模型后端"""
    LLAMA = "llama"        # 本地llama.cpp
    OLLAMA = "ollama"      # 本地ollama
    OPENAI = "openai"      # OpenAI兼容API（远程）


class LLMResponse:
    """大模型响应类"""
    def __init__(self, text: str, model: str, usage: Dict = None,
                 elapsed: float = 0.0, raw_response: Dict = None,
                 tool_calls: List = None):
        self.text = text
        self.model = model
        self.usage = usage or {}
        self.elapsed = elapsed  # 请求耗时（秒）
        self.raw_response = raw_response
        # function calling：模型请求的工具调用，统一为
        # [{"id": str, "name": str, "arguments": dict}]；无调用时为 None
        self.tool_calls = tool_calls

    def __str__(self):
        return f"LLMResponse(text='{self.text[:50]}...', model={self.model}, elapsed={self.elapsed:.2f}s)"

    def __repr__(self):
        return self.__str__()


class BaseLLMEndpoint(ABC):
    """大模型端点基类"""

    def __init__(self, base_url: str, model: str, timeout: int = 120,
                 max_tokens: int = 2048, temperature: float = 0.7):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成文本"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass

    def warmup(self, keep_alive: str = "30m") -> float:
        """预热：把模型加载进服务端内存（默认no-op，懒加载后端才实现）

        Returns:
            预热耗时（秒）；不支持时返回0.0。失败抛异常由调用方决定是否容忍。
        """
        return 0.0

    def _parse_choices(self, result: Dict, elapsed: float) -> LLMResponse:
        """解析OpenAI choices格式的响应（llama.cpp/OpenAI共用，含tool_calls）"""
        message = result["choices"][0]["message"]
        text = message.get("content") or ""
        usage = result.get("usage", {})

        # function calling：OpenAI格式arguments是JSON字符串，统一转为dict
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                args = tc["function"].get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc["function"]["name"],
                    "arguments": args,
                })

        return LLMResponse(
            text=text,
            model=self.model,
            usage=usage,
            elapsed=elapsed,
            raw_response=result,
            tool_calls=tool_calls
        )

    def _openai_chat_stream(self, messages: List[Dict], tools: List = None,
                            extra_headers: Dict = None, **kwargs):
        """OpenAI兼容SSE流式chat（llama.cpp/OpenAI端点共用）

        逐块产出事件dict：
          {"delta": str}                          增量文本
          {"done": True, "text": 全文, "usage":}  流结束（收尾统计）
        tool_calls以增量片段跨块到达，聚齐后在流末尾作为事件产出。

        消费方break或close()本生成器 → finally关闭HTTP连接 →
        服务端中止后续生成（ollama实测断连即取消，llama.cpp同机制）。
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        req = urllib.request.Request(
            f"{self.base_url}{self._completions_path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = "(无法读取错误响应体)"
            raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"连接失败: {e}") from e

        try:
            full = []
            tool_acc = {}   # index -> {"id","name","arguments"}片段聚合
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue   # SSE注释行/空行
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    full.append(delta["content"])
                    yield {"delta": delta["content"]}
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    acc = tool_acc.setdefault(
                        tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    acc["id"] += tc.get("id") or ""
                    acc["name"] += fn.get("name") or ""
                    acc["arguments"] += fn.get("arguments") or ""
            if tool_acc:
                calls = []
                for acc in tool_acc.values():
                    args = acc["arguments"] or "{}"
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                    calls.append({"id": acc["id"], "name": acc["name"],
                                  "arguments": args})
                yield {"tool_calls": calls}
                return
            yield {"done": True, "text": "".join(full), "usage": {}}
        finally:
            resp.close()


class LlamaCPPEndpoint(BaseLLMEndpoint):
    """llama.cpp端点"""

    _completions_path = "/v1/chat/completions"

    def __init__(self, base_url: str = "http://localhost:8080",
                 model: str = "qwen2.5-vl-3b", **kwargs):
        super().__init__(base_url, model, **kwargs)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """调用llama.cpp生成文本"""
        # 支持messages格式（与camera_vlm_monitor.py已验证的用法一致）
        messages = kwargs.get('messages') or [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stream": False,
        }
        tools = kwargs.get('tools')
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start

            if "choices" not in result or not result["choices"]:
                raise RuntimeError(f"无效响应: {result}")

            return self._parse_choices(result, elapsed)

        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = "(无法读取错误响应体)"
            raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"连接失败: {e}") from e

    def health_check(self) -> bool:
        """llama.cpp健康检查"""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat_stream(self, messages: List[Dict], tools: List = None, **kwargs):
        """流式chat（SSE），事件格式见 _openai_chat_stream"""
        return self._openai_chat_stream(messages, tools=tools, **kwargs)


class OllamaEndpoint(BaseLLMEndpoint):
    """Ollama端点"""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:3b", **kwargs):
        super().__init__(base_url, model, **kwargs)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """调用Ollama生成文本"""
        messages = kwargs.get('messages')
        if messages:
            # 使用chat API (支持多轮对话)
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                    "temperature": kwargs.get('temperature', self.temperature),
                }
            }
            endpoint = f"{self.base_url}/api/chat"
            response_field = "message"
        else:
            # 使用generate API (单次对话)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                    "temperature": kwargs.get('temperature', self.temperature),
                }
            }
            endpoint = f"{self.base_url}/api/generate"
            response_field = "response"

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start

            text = result[response_field]
            # Ollama返回的usage格式不同
            usage = {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
            }

            return LLMResponse(
                text=text,
                model=self.model,
                usage=usage,
                elapsed=elapsed,
                raw_response=result
            )

        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = "(无法读取错误响应体)"
            raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"连接失败: {e}") from e

    def warmup(self, keep_alive: str = "30m") -> float:
        """预热：触发ollama把模型加载进内存并常驻keep_alive

        ollama是懒加载：首次请求要把模型权重读入内存（qwen2.5:7b约
        数秒到十几秒），之后默认驻留5分钟。预热用一个最小请求
        （num_predict=1）提前完成加载，语音对话首响应不再吃这笔延迟。

        Args:
            keep_alive: 模型驻留时长（如"5m"/"30m"/"-1"永久）。
                之后每次真实请求都会刷新该计时。

        Returns:
            预热耗时（秒），含模型加载时间
        """
        payload = {
            "model": self.model,
            "prompt": "hi",
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_predict": 1, "temperature": 0.1},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            json.loads(resp.read().decode("utf-8"))   # 只关心状态，内容丢弃
        return time.time() - start

    def chat(self, messages: List[Dict], tools: List = None, **kwargs) -> LLMResponse:
        """Ollama专用chat接口（支持多轮对话与function calling）

        Args:
            messages: 消息列表（可含 role=tool 的工具结果消息）
            tools: OpenAI格式的工具schema列表，None表示不启用
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": kwargs.get('max_tokens', self.max_tokens),
                "temperature": kwargs.get('temperature', self.temperature),
            }
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start

            message = result["message"]
            text = message.get("content", "")
            usage = {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
            }

            # function calling：统一解析为 [{"id","name","arguments"(dict)}]
            tool_calls = None
            if message.get("tool_calls"):
                tool_calls = [
                    {
                        "id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments") or {},
                    }
                    for tc in message["tool_calls"]
                ]

            return LLMResponse(
                text=text,
                model=self.model,
                usage=usage,
                elapsed=elapsed,
                raw_response=result,
                tool_calls=tool_calls
            )

        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = "(无法读取错误响应体)"
            raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"连接失败: {e}") from e

    def health_check(self) -> bool:
        """Ollama健康检查"""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat_stream(self, messages: List[Dict], tools: List = None, **kwargs):
        """流式chat：NDJSON逐行事件生成器（R3流式响应）

        逐块产出事件dict：
          {"delta": str}                          增量文本
          {"tool_calls": [...]}                   工具调用决议（同chat()解析格式）
          {"done": True, "text": 全文, "usage":}  流结束

        实测（qwen2.5:7b）：文本增量每token一行；tool_calls是完整对象随
        某块到达（首块即到且content为空）——命中即产出并结束流。
        消费方break或close()本生成器 → finally关闭连接 → ollama实测
        断连即取消生成（后续请求0.13s响应，不会排队等旧生成烧完）。
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": kwargs.get('max_tokens', self.max_tokens),
                "temperature": kwargs.get('temperature', self.temperature),
            }
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = "(无法读取错误响应体)"
            raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"连接失败: {e}") from e

        try:
            full = []
            for raw in resp:
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                # 工具调用优先判定（与content互斥出现，qwen工具轮content为空）
                if msg.get("tool_calls"):
                    tool_calls = [
                        {
                            "id": tc.get("id", ""),
                            "name": tc["function"]["name"],
                            "arguments": tc["function"].get("arguments") or {},
                        }
                        for tc in msg["tool_calls"]
                    ]
                    yield {"tool_calls": tool_calls}
                    return   # 关闭连接 → 服务端取消生成
                content = msg.get("content") or ""
                if content:
                    full.append(content)
                    yield {"delta": content}
                if obj.get("done"):
                    usage = {
                        "prompt_tokens": obj.get("prompt_eval_count", 0),
                        "completion_tokens": obj.get("eval_count", 0),
                        "total_tokens": obj.get("eval_count", 0)
                                        + obj.get("prompt_eval_count", 0),
                    }
                    yield {"done": True, "text": "".join(full), "usage": usage}
                    return
            # 服务端未发done就断流：以已收内容收尾（防御）
            yield {"done": True, "text": "".join(full), "usage": {}}
        finally:
            resp.close()


class OpenAIEndpoint(BaseLLMEndpoint):
    """OpenAI兼容API端点（远程大模型）"""

    _completions_path = "/chat/completions"   # base_url已含/v1

    def __init__(self, base_url: str, model: str, api_key: str = None,
                 timeout: int = 120, **kwargs):
        super().__init__(base_url, model, timeout, **kwargs)
        self.api_key = api_key or ""

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """调用OpenAI兼容API生成文本"""
        messages = kwargs.get('messages', [{"role": "user", "content": prompt}])

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
        }
        tools = kwargs.get('tools')
        if tools:
            payload["tools"] = tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start

            if "choices" not in result or not result["choices"]:
                raise RuntimeError(f"无效响应: {result}")

            return self._parse_choices(result, elapsed)

        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = "(无法读取错误响应体)"
            raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"连接失败: {e}") from e

    def chat_stream(self, messages: List[Dict], tools: List = None, **kwargs):
        """流式chat（SSE），事件格式见 _openai_chat_stream"""
        headers = None
        if self.api_key:
            headers = {"Authorization": f"Bearer {self.api_key}"}
        return self._openai_chat_stream(messages, tools=tools,
                                        extra_headers=headers, **kwargs)

    def health_check(self) -> bool:
        """OpenAI兼容API健康检查"""
        try:
            # 尝试列出模型作为健康检查
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers=headers,
                method="GET",
            )

            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


class LLM:
    """统一大模型调用类"""

    # 默认配置
    DEFAULT_ENDPOINTS = {
        LLMBackend.LLAMA: {
            "base_url": "http://localhost:8080",
            "model": "qwen2.5-vl-3b",
            "timeout": 300,  # llama.cpp可能较慢
        },
        LLMBackend.OLLAMA: {
            "base_url": "http://localhost:11434",
            "model": "qwen2.5:7b",
        },
        LLMBackend.OPENAI: {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4",
            "api_key": None,
        }
    }

    def __init__(self, backend: str = "llama", base_url: str = None,
                 model: str = None, api_key: str = None, **kwargs):
        """
        初始化大模型接口

        Args:
            backend: 后端类型 ("llama", "ollama", "openai")
            base_url: 服务地址
            model: 模型名称
            api_key: API密钥（OpenAI兼容API需要）
            **kwargs: 其他参数（timeout, max_tokens, temperature等）
        """
        try:
            backend_enum = LLMBackend(backend)
        except ValueError:
            raise ValueError(f"不支持的后端: {backend}，支持: {[b.value for b in LLMBackend]}")

        # 使用默认配置或用户配置
        config = self.DEFAULT_ENDPOINTS[backend_enum].copy()
        if base_url:
            config["base_url"] = base_url
        if model:
            config["model"] = model
        if api_key:
            config["api_key"] = api_key
        config.update(kwargs)  # 用户参数覆盖默认值

        # 创建对应的后端实例
        if backend_enum == LLMBackend.LLAMA:
            self.endpoint = LlamaCPPEndpoint(**config)
        elif backend_enum == LLMBackend.OLLAMA:
            self.endpoint = OllamaEndpoint(**config)
        elif backend_enum == LLMBackend.OPENAI:
            if not config.get("api_key") and "api.openai.com" in config["base_url"]:
                raise ValueError("OpenAI需要api_key")
            self.endpoint = OpenAIEndpoint(**config)

        self.backend = backend_enum
        print(f"✅ {backend} 大模型接口初始化成功")
        print(f"📍 服务地址: {config['base_url']}")
        print(f"🎯 模型: {config['model']}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        生成文本

        Args:
            prompt: 输入提示词
            **kwargs: 额外参数（max_tokens, temperature等）

        Returns:
            LLMResponse对象，包含生成文本和元数据
        """
        return self.endpoint.generate(prompt, **kwargs)

    def health_check(self) -> bool:
        """健康检查"""
        return self.endpoint.health_check()

    def warmup(self, keep_alive: str = "30m") -> float:
        """预热模型（懒加载后端有效，其余no-op）。返回预热耗时秒"""
        return self.endpoint.warmup(keep_alive=keep_alive)

    def chat(self, messages: List[Dict], tools: List = None, **kwargs) -> LLMResponse:
        """
        对话模式（多轮对话，支持function calling）

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            tools: OpenAI格式的工具schema列表（可选）
            **kwargs: 额外参数

        Returns:
            LLMResponse对象（请求工具时 resp.tool_calls 非空，text为空）
        """
        if self.backend == LLMBackend.OLLAMA:
            # Ollama使用专用chat接口
            return self.endpoint.chat(messages, tools=tools, **kwargs)
        else:
            # OpenAI和llama.cpp使用标准messages格式
            return self.endpoint.generate("", messages=messages, tools=tools, **kwargs)

    def chat_stream(self, messages: List[Dict], tools: List = None, **kwargs):
        """流式对话（R3）：生成器逐块产出事件dict

          {"delta": str}                          增量文本
          {"tool_calls": [...]}                   工具调用决议（格式同chat()）
          {"done": True, "text": 全文, "usage":}  流结束统计

        用法:
            gen = llm.chat_stream(messages)
            for ev in gen:
                if "delta" in ev: ...
            # 提前break后务必 gen.close() → 关闭HTTP连接 → 服务端取消生成
        """
        return self.endpoint.chat_stream(messages, tools=tools, **kwargs)

    def _format_messages_to_prompt(self, messages: List[Dict]) -> str:
        """将消息格式转换为单prompt（Ollama使用）"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)

    def __call__(self, prompt: str, **kwargs) -> LLMResponse:
        """便捷调用方式：llm("你好")"""
        return self.generate(prompt, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='统一大模型调用工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 本地llama.cpp
  python unified_llm.py --backend llama --prompt "你好"

  # 本地ollama
  python unified_llm.py --backend ollama --prompt "介绍一下自己"

  # 远程OpenAI
  python unified_llm.py --backend openai --api-key YOUR_KEY --prompt "分析数据"
        '''
    )

    parser.add_argument('--backend', '-b', type=str, default='llama',
                       choices=['llama', 'ollama', 'openai'],
                       help='大模型后端（默认: llama）')
    parser.add_argument('--url', '-u', type=str,
                       help='服务地址（覆盖默认地址）')
    parser.add_argument('--model', '-m', type=str,
                       help='模型名称')
    parser.add_argument('--api-key', '-k', type=str,
                       help='API密钥（OpenAI兼容API需要）')
    parser.add_argument('--prompt', '-p', type=str, default='你好，请介绍一下自己',
                       help='输入提示词')
    parser.add_argument('--max-tokens', '-n', type=int, default=512,
                       help='最大生成token数')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                       help='温度参数')
    parser.add_argument('--health', action='store_true',
                       help='只进行健康检查')

    args = parser.parse_args()

    try:
        # 创建LLM实例
        llm = LLM(
            backend=args.backend,
            base_url=args.url,
            model=args.model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # 健康检查
        if args.health:
            is_healthy = llm.health_check()
            print(f"健康检查: {'✅ 服务正常' if is_healthy else '❌ 服务不可达'}")
            return

        # 生成文本
        print(f"\n💭 输入: {args.prompt}")
        print("🧠 正在生成...")
        result = llm(args.prompt)

        print(f"\n💬 输出: {result.text}")
        print(f"⏱️  耗时: {result.elapsed:.2f}s")
        if result.usage:
            print(f"📊 Token统计: {result.usage}")

    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()