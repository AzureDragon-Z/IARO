"""
异步LLM客户端模块 (llm_client.py)
封装 AsyncOpenAI，实现带指数退避的重试机制

用于ACL论文: Bridging the Intent Gap: Multi-Faceted Intent Recognition and Inference-Time Prompt Optimization
"""

import asyncio
import json
import time
import random
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError

from config import config, get_api_key, get_base_url


# ========== 请求统计 ==========
@dataclass
class RequestStats:
    """请求统计信息"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency: float = 0.0
    
    def record_success(self, tokens: Dict[str, int], latency: float):
        """记录成功请求"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_tokens += tokens.get("total_tokens", 0)
        self.prompt_tokens += tokens.get("prompt_tokens", 0)
        self.completion_tokens += tokens.get("completion_tokens", 0)
        self.total_latency += latency
    
    def record_failure(self):
        """记录失败请求"""
        self.total_requests += 1
        self.failed_requests += 1
    
    def record_retry(self):
        """记录重试"""
        self.retried_requests += 1
    
    @property
    def average_latency(self) -> float:
        """平均延迟"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retried_requests": self.retried_requests,
            "success_rate": f"{self.success_rate:.2%}",
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "average_latency_ms": f"{self.average_latency * 1000:.2f}",
        }


# ========== 异步LLM客户端 ==========
class AsyncLLMClient:
    """
    异步LLM客户端
    
    特性:
    - 支持 OpenAI 兼容的 API (OpenAI, DeepSeek, SiliconFlow)
    - 指数退避重试机制
    - 并发控制 (Semaphore)
    - 请求统计
    - JSON 模式支持
    """
    
    def __init__(
        self,
        provider: str = "siliconflow",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: int = 120,
        semaphore_limit: int = 10,
    ):
        """
        初始化异步LLM客户端
        
        Args:
            provider: API提供商 ("siliconflow", "deepseek", "openai")
            model: 模型名称，None则使用默认模型
            api_key: API密钥，None则从配置读取
            base_url: API地址，None则从配置读取
            max_retries: 最大重试次数
            base_delay: 基础重试延迟（秒）
            max_delay: 最大重试延迟（秒）
            timeout: 请求超时时间（秒）
            semaphore_limit: 并发限制
        """
        self.provider = provider
        self.model = model or config.get_provider(provider).default_model
        
        # API配置
        self.api_key = api_key or get_api_key(provider)
        self.base_url = base_url or get_base_url(provider)
        
        # 从 provider config 获取配置
        provider_config = config.get_provider(provider)
        self.max_context_length = provider_config.max_context_length if provider_config else 128000
        
        # 使用 provider 特定的 timeout 和 semaphore_limit (如果未显式指定)
        effective_timeout = timeout
        effective_semaphore = semaphore_limit
        if provider_config:
            # 如果使用默认值，则从 provider config 读取
            if timeout == 120:
                effective_timeout = provider_config.timeout
            if semaphore_limit == 10:
                effective_semaphore = provider_config.semaphore_limit
        
        if not self.api_key:
            raise ValueError(f"API key not found for provider: {provider}")
        
        # 重试配置
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = effective_timeout
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(effective_semaphore)
        
        # 统计信息
        self.stats = RequestStats()
        
        # 创建异步客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        
        print(f"✓ AsyncLLMClient initialized: {provider} / {self.model}")
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        计算指数退避延迟
        
        使用 exponential backoff with jitter:
        delay = min(base_delay * 2^attempt + random_jitter, max_delay)
        """
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        # 添加随机抖动 (±25%)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)
    
    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行单次API请求（带重试）
        
        Args:
            messages: 消息列表
            temperature: 采样温度
            max_tokens: 最大输出token数
            json_mode: 是否启用JSON模式
            **kwargs: 其他参数
        
        Returns:
            包含响应内容和元数据的字典
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # 构建请求参数
                request_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                # JSON模式
                if json_mode:
                    request_params["response_format"] = {"type": "json_object"}
                
                # 合并额外参数
                request_params.update(kwargs)
                
                # 发起请求
                response = await self.client.chat.completions.create(**request_params)
                
                latency = time.time() - start_time
                
                # 提取响应内容
                content = response.choices[0].message.content
                
                # 统计token使用
                tokens = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                
                # 记录成功
                self.stats.record_success(tokens, latency)
                
                return {
                    "content": content,
                    "tokens": tokens,
                    "latency": latency,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                }
                
            except RateLimitError as e:
                # 速率限制 - 等待后重试
                delay = self._calculate_delay(attempt)
                print(f"⚠ Rate limit hit, waiting {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                self.stats.record_retry()
                await asyncio.sleep(delay)
                last_exception = e
                
            except APITimeoutError as e:
                # 超时 - 重试
                delay = self._calculate_delay(attempt)
                print(f"⚠ Request timeout, retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                self.stats.record_retry()
                await asyncio.sleep(delay)
                last_exception = e
                
            except APIConnectionError as e:
                # 连接错误 - 重试
                delay = self._calculate_delay(attempt)
                print(f"⚠ Connection error, retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                self.stats.record_retry()
                await asyncio.sleep(delay)
                last_exception = e
                
            except APIError as e:
                # 其他API错误
                if e.status_code in [500, 502, 503, 504]:
                    # 服务器错误 - 重试
                    delay = self._calculate_delay(attempt)
                    print(f"⚠ Server error ({e.status_code}), retrying in {delay:.2f}s")
                    self.stats.record_retry()
                    await asyncio.sleep(delay)
                    last_exception = e
                else:
                    # 其他错误 - 不重试
                    self.stats.record_failure()
                    raise
                    
            except Exception as e:
                # 未知错误
                self.stats.record_failure()
                raise
        
        # 所有重试都失败
        self.stats.record_failure()
        raise last_exception or Exception("Max retries exceeded")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        对于 Llama 等模型，使用保守估计: 1 token ≈ 2.5 字符 (中英文混合更保守)
        """
        # 更保守的估计，确保不会超出限制
        return int(len(text) / 2.5) + 50
    
    def _adjust_max_tokens(self, messages: list, requested_max_tokens: int) -> int:
        """
        根据上下文长度限制动态调整max_tokens
        
        Args:
            messages: 消息列表
            requested_max_tokens: 请求的max_tokens
        
        Returns:
            调整后的max_tokens
        """
        # 估算输入token数
        input_text = "".join(m.get("content", "") for m in messages)
        estimated_input_tokens = self._estimate_tokens(input_text)
        
        # 计算可用的输出token数 (留300 token安全余量)
        available_tokens = self.max_context_length - estimated_input_tokens - 300
        
        # 取较小值，最少保留500 tokens用于输出
        adjusted_max_tokens = min(requested_max_tokens, max(500, available_tokens))
        
        if adjusted_max_tokens < requested_max_tokens:
            print(f"  ⚠ Adjusted max_tokens: {requested_max_tokens} -> {adjusted_max_tokens} (context limit: {self.max_context_length})")
        
        return adjusted_max_tokens
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 采样温度
            max_tokens: 最大输出token数
            json_mode: 是否启用JSON模式
        
        Returns:
            生成的文本内容
        """
        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 动态调整max_tokens (针对上下文长度较小的模型)
        adjusted_max_tokens = self._adjust_max_tokens(messages, max_tokens)
        
        # 使用信号量控制并发
        async with self.semaphore:
            result = await self._make_request(
                messages=messages,
                temperature=temperature,
                max_tokens=adjusted_max_tokens,
                json_mode=json_mode,
                **kwargs
            )
        
        return result["content"]
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成JSON格式响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 采样温度（JSON模式建议使用较低温度）
            max_tokens: 最大输出token数
        
        Returns:
            解析后的JSON字典
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            content = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                **kwargs
            )
            
            # 解析JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                # 尝试提取JSON部分
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
                
                last_error = e
                if attempt < max_retries - 1:
                    print(f"⚠ JSON parse failed (attempt {attempt + 1}/{max_retries}), retrying...")
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    print(f"⚠ Failed to parse JSON after {max_retries} attempts: {e}")
                    print(f"  Content: {content[:500]}...")
        
        raise ValueError(f"Invalid JSON response: {last_error}")
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        批量生成文本响应
        
        Args:
            prompts: 提示列表
            system_prompt: 系统提示
            temperature: 采样温度
            max_tokens: 最大输出token数
            show_progress: 是否显示进度条
        
        Returns:
            生成的文本列表
        """
        from tqdm.asyncio import tqdm_asyncio
        
        async def generate_one(prompt: str) -> str:
            return await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        tasks = [generate_one(p) for p in prompts]
        
        if show_progress:
            results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"Generating ({self.model})",
                total=len(tasks)
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"⚠ Task {i} failed: {r}")
                processed_results.append("")
            else:
                processed_results.append(r)
        
        return processed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider": self.provider,
            "model": self.model,
            **self.stats.to_dict()
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = RequestStats()


# ========== 便捷工厂函数 ==========
def create_client(
    provider: str = "siliconflow",
    model: Optional[str] = None,
    **kwargs
) -> AsyncLLMClient:
    """
    创建LLM客户端
    
    Args:
        provider: 提供商名称
        model: 模型名称
        **kwargs: 其他参数
    
    Returns:
        AsyncLLMClient实例
    """
    return AsyncLLMClient(provider=provider, model=model, **kwargs)


def create_solver_client() -> AsyncLLMClient:
    """创建用于求解的客户端"""
    return AsyncLLMClient(
        provider=config.model_roles.solver,
        model=config.model_roles.solver_model,
    )


def create_judge_client() -> AsyncLLMClient:
    """创建用于评判的客户端"""
    return AsyncLLMClient(
        provider=config.model_roles.judge,
        model=config.model_roles.judge_model,
    )


def create_generator_client() -> AsyncLLMClient:
    """创建用于数据生成的客户端"""
    return AsyncLLMClient(
        provider=config.model_roles.data_generator,
        model=config.model_roles.data_generator_model,
    )


# ========== 测试代码 ==========
async def test_client():
    """测试客户端"""
    print("=" * 50)
    print("Testing AsyncLLMClient")
    print("=" * 50)
    
    client = create_client("siliconflow")
    
    # 测试单次生成
    print("\n1. Testing single generation...")
    response = await client.generate(
        prompt="What is 2 + 2? Answer briefly.",
        temperature=0.1,
        max_tokens=50
    )
    print(f"Response: {response}")
    
    # 测试JSON生成
    print("\n2. Testing JSON generation...")
    json_response = await client.generate_json(
        prompt="List 3 colors in JSON format with keys 'colors' as an array.",
        system_prompt="You are a helpful assistant that outputs valid JSON.",
        temperature=0.1,
    )
    print(f"JSON Response: {json_response}")
    
    # 打印统计
    print("\n3. Statistics:")
    print(json.dumps(client.get_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(test_client())
