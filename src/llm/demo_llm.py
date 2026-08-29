#!/usr/bin/env python3
"""统一大模型类使用演示"""
import sys
sys.path.insert(0, 'src/llm')
from unified_llm import LLM

def demo_basic_usage():
    """基础使用演示"""
    print("🎯 统一大模型类 - 基础使用演示")
    print("=" * 60)
    print()

    print("📦 1. 创建LLM实例（不同后端）")
    print("-" * 40)

    # llama.cpp
    print("🔸 llama.cpp:")
    llama_llm = LLM(
        backend='llama',
        base_url='http://localhost:8080',
        model='qwen2.5-vl-3b',
        timeout=300
    )
    print()

    # ollama
    print("🔸 ollama:")
    ollama_llm = LLM(
        backend='ollama',
        base_url='http://localhost:11434',
        model='qwen2.5:7b'
    )
    print()

    # openai
    print("🔸 OpenAI兼容:")
    openai_llm = LLM(
        backend='openai',
        base_url='https://api.openai.com/v1',
        model='gpt-4',
        api_key='your-api-key'
    )
    print()

def demo_programming_usage():
    """编程使用演示"""
    print("🎯 编程使用演示")
    print("=" * 60)
    print()

    print("📦 2. 编程方式调用")
    print("-" * 40)

    # 示例代码展示
    code_examples = """
# 基础使用
from unified_llm import LLM

# 创建实例
llm = LLM(backend='llama', base_url='http://localhost:8080')

# 方式1: 直接调用
result = llm("你好，请介绍一下自己")
print(result.text)

# 方式2: generate方法
result = llm.generate("写一首关于春天的诗")
print(f"生成: {result.text}")
print(f"耗时: {result.elapsed}s")

# 方式3: 对话模式
messages = [
    {"role": "system", "content": "你是一个编程助手"},
    {"role": "user", "content": "解释Python的装饰器"}
]
result = llm.chat(messages)
print(result.text)

# 方式4: 自定义参数
result = llm.generate("分析这个数据",
                       max_tokens=1000,
                       temperature=0.3)
"""

    print(code_examples)
    print()

def demo_advanced_features():
    """高级功能演示"""
    print("🎯 高级功能演示")
    print("=" * 60)
    print()

    print("📦 3. 高级功能")
    print("-" * 40)

    features = """
✨ 统一接口:
   - 支持3种后端，API完全一致
   - 切换后端只需改backend参数

✨ 智能配置:
   - 每个后端有合理的默认配置
   - 用户参数覆盖默认值

✨ 响应对象:
   - 返回LLMResponse对象，包含丰富元数据
   - text: 生成文本
   - model: 使用的模型
   - usage: token统计
   - elapsed: 请求耗时

✨ 健康检查:
   - llm.health_check() 检查服务可用性

✨ 错误处理:
   - 统一的异常处理
   - 清晰的错误信息
"""

    print(features)
    print()

def demo_comparison():
    """后端对比演示"""
    print("🎯 三种后端对比")
    print("=" * 60)
    print()

    comparison = """
┌─────────────┬──────────────┬─────────────┬──────────────┐
│    特性     │   llama.cpp  │   ollama    │   OpenAI     │
├─────────────┼──────────────┼─────────────┼──────────────┤
│   部署方式   │    本地       │    本地      │    远程      │
│   模型管理   │   手动下载    │   自动拉取   │   托管        │
│   配置难度   │    中等      │    简单      │    简单      │
│   性能       │   依赖硬件   │   较好       │   最快       │
│   成本       │    免费      │    免费      │   按量付费    │
│   推荐场景   │  离线部署   │  本地开发   │  生产环境    │
└─────────────┴──────────────┴─────────────┴──────────────┘
"""

    print(comparison)
    print()

def demo_use_cases():
    """使用场景演示"""
    print("🎯 使用场景")
    print("=" * 60)
    print()

    use_cases = """
📚 推荐使用场景:

1. 本地开发和测试:
   - ollama: 模型管理简单，支持多种模型
   - 命令: LLM(backend='ollama')

2. 离线部署和嵌入式:
   - llama.cpp: 轻量级，资源占用少
   - 命令: LLM(backend='llama', base_url='http://localhost:8080')

3. 生产环境和云计算:
   - OpenAI: 高性能，高可用性
   - 命令: LLM(backend='openai', api_key='your-key')

4. 多模型切换测试:
   - 相同代码，不同后端
   - 只需修改backend参数

5. 原型开发和演示:
   - 统一接口便于快速迭代
   - 健康检查确保服务可用
"""

    print(use_cases)
    print()

def main():
    print("🎭 统一大模型类 - 完整演示")
    print("=" * 60)
    print()

    demo_basic_usage()
    demo_programming_usage()
    demo_advanced_features()
    demo_comparison()
    demo_use_cases()

    print("🎉 演示完成！")
    print()
    print("📋 快速开始:")
    print("  # 导入类")
    print("  from unified_llm import LLM")
    print()
    print("  # 创建实例")
    print("  llm = LLM(backend='llama', base_url='http://localhost:8080')")
    print()
    print("  # 生成文本")
    print("  result = llm('你好，请介绍一下自己')")
    print("  print(result.text)")

if __name__ == "__main__":
    main()
