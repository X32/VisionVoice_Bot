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


class LlamaCPPEndpoint(BaseLLMEndpoint):
    """llama.cpp端点"""

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


class OpenAIEndpoint(BaseLLMEndpoint):
    """OpenAI兼容API端点（远程大模型）"""

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