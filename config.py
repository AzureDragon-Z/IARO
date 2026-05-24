"""
API配置模块 (config.py)
管理 API Keys, Base URLs, Model Names, 并发控制等

用于ACL论文: Bridging the Intent Gap: Multi-Faceted Intent Recognition and Inference-Time Prompt Optimization
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# ========== 环境变量加载 ==========
def load_environment():
    """加载环境变量，优先使用 .env.production"""
    project_root = Path(__file__).parent.parent.parent
    
    env_paths = [
        project_root / '.env.production',
        project_root / '.env',
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            print(f"✓ Loaded environment from: {env_path}")
            return
    
    print("⚠ Warning: No .env file found, using system environment variables")

# 初始化时加载环境变量
load_environment()


# ========== API 提供商配置 ==========
@dataclass
class ProviderConfig:
    """API提供商配置"""
    name: str
    api_key: str
    base_url: str
    default_model: str
    
    # 速率限制
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    
    # 上下文长度限制 (默认128k, 本地模型可能较小)
    max_context_length: int = 128000
    
    # 超时和并发配置 (本地模型需要更长超时和更低并发)
    timeout: int = 120
    semaphore_limit: int = 10
    
    # 支持的模型列表
    supported_models: list = field(default_factory=list)


# ========== 预定义提供商配置 ==========
PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "siliconflow": ProviderConfig(
        name="SiliconFlow",
        api_key=os.getenv("SILICONFLOW_API_KEY", ""),
        base_url="https://api.siliconflow.cn/v1",
        default_model="deepseek-ai/DeepSeek-V3",
        requests_per_minute=100,
        tokens_per_minute=200000,
        timeout=900,              # 15分钟超时 (IARO多轮调用需要更长时间)
        semaphore_limit=2,        # 低并发避免API过载
        supported_models=[
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-V2.5",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
        ]
    ),
    "deepseek": ProviderConfig(
        name="DeepSeek",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        requests_per_minute=60,
        tokens_per_minute=100000,
        supported_models=[
            "deepseek-chat",
            "deepseek-reasoner",
        ]
    ),
    "openai": ProviderConfig(
        name="OpenAI",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        requests_per_minute=60,
        tokens_per_minute=150000,
        supported_models=[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
    ),
    "local": ProviderConfig(
        name="Local",
        api_key=os.getenv("LOCAL_API_KEY", "EMPTY"),  # vLLM doesn't need real API key
        base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1"),
        default_model="/user/zql/model/Llama-2-13b-chat-hf",
        requests_per_minute=100,
        tokens_per_minute=500000,
        max_context_length=4096,  # Llama-2-13b 的上下文窗口
        timeout=600,              # 10分钟超时 (本地模型推理慢)
        semaphore_limit=2,        # 低并发 (单GPU无法处理高并发)
        supported_models=[
            "/user/zql/model/Llama-2-13b-chat-hf",
        ]
    ),
}


# ========== 实验配置 ==========
@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据集配置
    dataset_name: str = "AmbiguBench-SOTA"
    dataset_version: str = "v1.0"
    data_path: str = "data/ambigubench"
    
    # 生成数据量
    num_samples_train: int = 500
    num_samples_test: int = 100
    
    # 领域分布 (确保泛化性，含新增高风险领域)
    domains: Dict[str, float] = field(default_factory=lambda: {
        "coding_python": 0.18,
        "coding_cpp": 0.08,
        "security_ops": 0.15,       # 新增：高风险运维场景
        "data_analysis": 0.15,      # 新增：高隐式需求场景
        "writing_email": 0.12,
        "writing_report": 0.10,
        "planning_travel": 0.10,
        "planning_project": 0.06,
        "qa_technical": 0.06,
    })
    
    # 并发控制 (降低并发以适应IARO多轮调用)
    max_concurrent_requests: int = 3
    semaphore_limit: int = 3
    
    # 重试配置
    max_retries: int = 5
    base_retry_delay: float = 2.0
    max_retry_delay: float = 120.0
    
    # 超时配置 (IARO多轮调用需要更长时间)
    request_timeout: int = 600
    
    # 随机种子
    random_seed: int = 42
    
    # 输出配置
    output_dir: str = "results/api_experiments"
    save_intermediate: bool = True
    
    # 日志配置
    log_level: str = "INFO"


# ========== 模型角色配置 ==========
@dataclass
class ModelRoleConfig:
    """模型角色配置 - 用于不同任务的模型选择"""
    # 数据生成模型 (需要强大的指令遵循能力)
    data_generator: str = "siliconflow"  # 使用 DeepSeek V3 via SiliconFlow
    data_generator_model: str = "deepseek-ai/DeepSeek-V3"
    
    # 质检员模型 (Self-Correction过滤)
    quality_checker: str = "siliconflow"
    quality_checker_model: str = "deepseek-ai/DeepSeek-V3"
    
    # 被测模型 (实验主体)
    solver: str = "siliconflow"
    solver_model: str = "deepseek-ai/DeepSeek-V3"
    
    # 评判模型 (Cross-Model Evaluation)
    # 使用 DeepSeek-V3 作为 Judge (速度快，能力足够)
    judge: str = "siliconflow"
    judge_model: str = "deepseek-ai/DeepSeek-V3"
    
    # 备用评判模型
    judge_alternative: str = "openai"
    judge_alternative_model: str = "gpt-4o"


# ========== 评估指标配置 ==========
@dataclass
class EvaluationConfig:
    """评估配置"""
    # 核心指标
    primary_metric: str = "ICR"  # Implicit Constraint Recall
    
    # 辅助指标
    secondary_metrics: list = field(default_factory=lambda: [
        "goal_completion_rate",
        "style_adherence",
        "response_quality",
    ])
    
    # 评判温度 (低温度确保一致性)
    judge_temperature: float = 0.1
    
    # 统计检验
    statistical_tests: list = field(default_factory=lambda: [
        "paired_t_test",
        "wilcoxon_signed_rank",
        "cohens_d",
    ])
    
    # 置信水平
    confidence_level: float = 0.95


# ========== 全局配置实例 ==========
class Config:
    """全局配置管理器"""
    
    def __init__(self):
        self.experiment = ExperimentConfig()
        self.model_roles = ModelRoleConfig()
        self.evaluation = EvaluationConfig()
        self.providers = PROVIDER_CONFIGS
    
    def get_provider(self, name: str) -> ProviderConfig:
        """获取提供商配置"""
        if name not in self.providers:
            raise ValueError(f"Unknown provider: {name}. Available: {list(self.providers.keys())}")
        return self.providers[name]
    
    def validate(self) -> bool:
        """验证配置有效性"""
        errors = []
        
        # 检查必要的API Key
        if not self.providers["siliconflow"].api_key:
            errors.append("SILICONFLOW_API_KEY not set")
        
        if self.model_roles.judge == "openai" and not self.providers["openai"].api_key:
            errors.append("OPENAI_API_KEY not set (required for judge)")
        
        if errors:
            print("❌ Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            return False
        
        print("✓ Configuration validated successfully")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """导出配置为字典"""
        return {
            "experiment": self.experiment.__dict__,
            "model_roles": self.model_roles.__dict__,
            "evaluation": self.evaluation.__dict__,
        }


# ========== 全局配置实例 ==========
config = Config()


# ========== 便捷访问函数 ==========
def get_api_key(provider: str) -> str:
    """获取API Key"""
    return config.get_provider(provider).api_key


def get_base_url(provider: str) -> str:
    """获取Base URL"""
    return config.get_provider(provider).base_url


def get_default_model(provider: str) -> str:
    """获取默认模型"""
    return config.get_provider(provider).default_model


if __name__ == "__main__":
    # 测试配置
    print("=" * 50)
    print("API Framework Configuration")
    print("=" * 50)
    
    config.validate()
    
    print("\nProvider Status:")
    for name, provider in config.providers.items():
        key_status = "✓" if provider.api_key else "✗"
        print(f"  {name}: {key_status} (model: {provider.default_model})")
    
    print("\nModel Roles:")
    print(f"  Data Generator: {config.model_roles.data_generator_model}")
    print(f"  Solver: {config.model_roles.solver_model}")
    print(f"  Judge: {config.model_roles.judge_model}")
