"""
数据集适配器模块 (benchmark_adapter.py)
为SST5、RACE、MedMCQA、AlpacaEval数据集提供统一的加载和格式化接口

用于ACL论文: Bridging the Intent Gap
"""

import json
import csv
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re


# ========== 统一数据样本结构 ==========
@dataclass
class BenchmarkSample:
    """基准测试样本的统一结构"""
    id: str
    dataset: str  # "sst5", "race", "medmcqa", "alpaca_eval"
    
    # 输入
    text: str  # 主要文本 (SST5的句子, RACE的问题, MedMCQA的问题)
    context: str = ""  # 上下文 (RACE的文章)
    options: List[str] = field(default_factory=list)  # 选项 (MCQA)
    
    # 标签
    label: str = ""  # 标准答案
    label_id: int = -1  # 数字标签
    
    # 元数据
    difficulty: str = "medium"  # easy, medium, hard
    subdomain: str = ""  # 子领域
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSample":
        return cls(**data)


# ========== 抽象适配器基类 ==========
class BaseAdapter(ABC):
    """数据集适配器基类"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples: List[BenchmarkSample] = []
    
    @abstractmethod
    def load(self, split: str = "dev", max_samples: int = None, seed: int = 42) -> List[BenchmarkSample]:
        """加载数据集"""
        pass
    
    @abstractmethod
    def get_task_type(self) -> str:
        """返回任务类型: classification, mcqa"""
        pass
    
    @abstractmethod
    def get_label_list(self) -> List[str]:
        """返回标签列表"""
        pass
    
    def get_prompt_template(self) -> str:
        """返回任务的提示模板"""
        return ""


# ========== SST5 适配器 ==========
class SST5Adapter(BaseAdapter):
    """SST-5 情感分析数据集适配器"""
    
    LABEL_MAP = {
        0: 'very negative',
        1: 'negative',
        2: 'neutral',
        3: 'positive',
        4: 'very positive'
    }
    
    LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # 默认路径
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "sst5"
        super().__init__(data_dir)
    
    def get_task_type(self) -> str:
        return "classification"
    
    def get_label_list(self) -> List[str]:
        return list(self.LABEL_MAP.values())
    
    def load(self, split: str = "dev", max_samples: int = None, seed: int = 42) -> List[BenchmarkSample]:
        """加载SST-5数据集"""
        data_file = self.data_dir / f"{split}.jsonl"
        
        if not data_file.exists():
            # 尝试其他可能的文件名
            alternatives = [
                self.data_dir / f"sst5_{split}.jsonl",
                self.data_dir / f"{split}.json",
            ]
            for alt in alternatives:
                if alt.exists():
                    data_file = alt
                    break
        
        if not data_file.exists():
            print(f"⚠ SST5 data file not found: {data_file}")
            print(f"  Available files: {list(self.data_dir.glob('*'))}")
            return []
        
        samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    text = item.get('text') or item.get('sentence') or item.get('review', '')
                    label = item.get('label')
                    
                    if text and label is not None:
                        if isinstance(label, int):
                            label_text = self.LABEL_MAP.get(label, 'neutral')
                            label_id = label
                        else:
                            label_text = label.lower()
                            label_id = self.LABEL_TO_ID.get(label_text, 2)
                        
                        # 评估难度
                        difficulty = self._estimate_difficulty(text)
                        
                        samples.append(BenchmarkSample(
                            id=f"sst5_{split}_{idx}",
                            dataset="sst5",
                            text=text,
                            label=label_text,
                            label_id=label_id,
                            difficulty=difficulty,
                            subdomain="sentiment",
                            options=list(self.LABEL_MAP.values()),
                        ))
                except json.JSONDecodeError:
                    continue
        
        print(f"✓ Loaded {len(samples)} SST5 samples from {split}")
        
        # 随机采样
        if max_samples and len(samples) > max_samples:
            random.seed(seed)
            samples = random.sample(samples, max_samples)
            print(f"  Randomly selected {max_samples} samples (seed={seed})")
        
        self.samples = samples
        return samples
    
    def _estimate_difficulty(self, text: str) -> str:
        """估计样本难度"""
        # 简单启发式: 短文本或包含明确情感词 -> easy
        # 中等长度 -> medium
        # 长文本或包含对比词 -> hard
        
        text_lower = text.lower()
        
        # 检测可能的讽刺/对比
        contrast_words = ['but', 'however', 'although', 'despite', 'yet']
        has_contrast = any(w in text_lower for w in contrast_words)
        
        # 检测明确情感词
        strong_positive = ['excellent', 'amazing', 'wonderful', 'fantastic', 'love']
        strong_negative = ['terrible', 'awful', 'horrible', 'hate', 'worst']
        has_strong = any(w in text_lower for w in strong_positive + strong_negative)
        
        word_count = len(text.split())
        
        if has_contrast or word_count > 30:
            return "hard"
        elif has_strong and word_count < 15:
            return "easy"
        else:
            return "medium"
    
    def get_prompt_template(self) -> str:
        return """请对以下文本进行5分类情感分析。

文本: {text}

请从以下类别中选择最合适的情感标签:
- very negative: 强烈的负面情感 (愤怒、厌恶)
- negative: 负面情感 (不满、失望)
- neutral: 中性/无明显情感
- positive: 正面情感 (满意、喜欢)
- very positive: 强烈的正面情感 (热情、极度满意)

请直接输出分类结果: sentiment: <类别>"""


# ========== RACE 适配器 ==========
class RACEAdapter(BaseAdapter):
    """RACE 阅读理解数据集适配器"""
    
    ANSWER_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "race"
        super().__init__(data_dir)
    
    def get_task_type(self) -> str:
        return "mcqa"
    
    def get_label_list(self) -> List[str]:
        return ['A', 'B', 'C', 'D']
    
    def load(self, split: str = "dev", max_samples: int = None, seed: int = 42,
             level: str = "high") -> List[BenchmarkSample]:
        """
        加载RACE数据集
        
        Args:
            split: 数据集划分 (dev, test, train)
            max_samples: 最大样本数
            seed: 随机种子
            level: 难度级别 (high, middle)
        """
        samples = []
        
        # 尝试加载 parquet 格式 (Hugging Face 下载的格式)
        parquet_split = 'validation' if split == 'dev' else split
        parquet_file = self.data_dir / level / f"{parquet_split}-00000-of-00001.parquet"
        
        if parquet_file.exists():
            samples = self._load_from_parquet(parquet_file, level, split)
        else:
            # 回退到原始 txt/json 格式
            samples = self._load_from_txt(level, split)
        
        if not samples:
            print(f"⚠ No samples loaded for RACE {level}/{split}")
            return []
        
        print(f"✓ Loaded {len(samples)} RACE samples from {level}/{split}")
        
        if max_samples and len(samples) > max_samples:
            random.seed(seed)
            samples = random.sample(samples, max_samples)
            print(f"  Randomly selected {max_samples} samples (seed={seed})")
        
        self.samples = samples
        return samples
    
    def _load_from_parquet(self, parquet_file: Path, level: str, split: str) -> List[BenchmarkSample]:
        """从 parquet 文件加载"""
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_file)
        except ImportError:
            print("⚠ pandas/pyarrow not installed, cannot read parquet")
            return []
        except Exception as e:
            print(f"⚠ Error reading parquet: {e}")
            return []
        
        samples = []
        for idx, row in df.iterrows():
            article = row.get('article', '')
            question = row.get('question', '')
            options = row.get('options', [])
            answer = row.get('answer', 'A')
            
            # 分析问题类型
            q_type = self._classify_question_type(question)
            
            samples.append(BenchmarkSample(
                id=f"race_{level}_{split}_{idx}",
                dataset="race",
                text=question,
                context=article,
                options=list(options) if hasattr(options, '__iter__') else [],
                label=answer,
                label_id=self.ANSWER_MAP.get(answer, 0),
                difficulty="hard" if level == "high" else "medium",
                subdomain=q_type,
                metadata={"level": level}
            ))
        
        return samples
    
    def _load_from_txt(self, level: str, split: str) -> List[BenchmarkSample]:
        """从 txt/json 文件加载 (原始格式)"""
        data_path = self.data_dir / level / split
        
        if not data_path.exists():
            alt_path = self.data_dir / level / ('dev' if split == 'validation' else split)
            if alt_path.exists():
                data_path = alt_path
            else:
                return []
        
        samples = []
        sample_idx = 0
        
        for file_path in data_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                article = data.get('article', '')
                questions = data.get('questions', [])
                options_list = data.get('options', [])
                answers = data.get('answers', [])
                
                for q, opts, ans in zip(questions, options_list, answers):
                    q_type = self._classify_question_type(q)
                    
                    samples.append(BenchmarkSample(
                        id=f"race_{level}_{split}_{sample_idx}",
                        dataset="race",
                        text=q,
                        context=article,
                        options=opts,
                        label=ans,
                        label_id=self.ANSWER_MAP.get(ans, 0),
                        difficulty="hard" if level == "high" else "medium",
                        subdomain=q_type,
                        metadata={"level": level, "file": file_path.name}
                    ))
                    sample_idx += 1
                    
            except Exception as e:
                continue
        
        return samples
    
    def _classify_question_type(self, question: str) -> str:
        """分类问题类型"""
        q_lower = question.lower()
        
        if any(w in q_lower for w in ['what is', 'who is', 'when', 'where', 'how many']):
            return "factual"
        elif any(w in q_lower for w in ['why', 'how does', 'what can we infer', 'implies']):
            return "inference"
        elif any(w in q_lower for w in ['best title', 'main idea', 'purpose', 'conclusion']):
            return "reasoning"
        else:
            return "other"
    
    def get_prompt_template(self) -> str:
        return """请阅读以下文章，然后回答问题。

文章:
{context}

问题: {text}

选项:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

请选择最正确的答案，直接输出: answer: A/B/C/D"""


# ========== MedMCQA 适配器 ==========
class MedMCQAAdapter(BaseAdapter):
    """MedMCQA 医学多选题数据集适配器"""
    
    ANSWER_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    ANSWER_TO_ID = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "medmcqa"
        super().__init__(data_dir)
    
    def get_task_type(self) -> str:
        return "mcqa"
    
    def get_label_list(self) -> List[str]:
        return ['A', 'B', 'C', 'D']
    
    def load(self, split: str = "dev", max_samples: int = None, seed: int = 42) -> List[BenchmarkSample]:
        """加载MedMCQA数据集"""
        samples = []
        
        # 方式1: 尝试parquet格式
        parquet_dir = self.data_dir / "data"
        split_name = "validation" if split == "dev" else split
        
        if parquet_dir.exists():
            try:
                import pyarrow.parquet as pq
                parquet_files = list(parquet_dir.glob(f"{split_name}-*.parquet"))
                
                for pf in parquet_files:
                    table = pq.read_table(pf)
                    df = table.to_pandas()
                    
                    for idx, row in df.iterrows():
                        question = str(row.get('question', ''))
                        options = [
                            str(row.get('opa', '')),
                            str(row.get('opb', '')),
                            str(row.get('opc', '')),
                            str(row.get('opd', ''))
                        ]
                        answer_idx = row.get('cop', 0)
                        answer = self.ANSWER_MAP.get(int(answer_idx), 'A')
                        subject = str(row.get('subject_name', 'unknown'))
                        
                        if question and any(options):
                            samples.append(BenchmarkSample(
                                id=f"medmcqa_{split}_{len(samples)}",
                                dataset="medmcqa",
                                text=question,
                                options=options,
                                label=answer,
                                label_id=int(answer_idx) if answer_idx is not None else 0,
                                difficulty=self._estimate_difficulty(question, options),
                                subdomain=subject,
                                metadata={"subject": subject}
                            ))
                
                if samples:
                    print(f"✓ Loaded {len(samples)} MedMCQA samples from parquet ({split_name})")
                    
            except ImportError:
                print("  ⚠ pyarrow not installed, trying JSON format...")
            except Exception as e:
                print(f"  ⚠ Error loading parquet: {e}")
        
        # 方式2: 尝试JSON/JSONL格式
        if not samples:
            possible_files = [
                self.data_dir / f"{split}.json",
                self.data_dir / f"{split}.jsonl",
                self.data_dir / f"medmcqa_{split}.json",
                self.data_dir / "dev.json" if split == "validation" else None,
            ]
            
            for data_file in possible_files:
                if data_file and data_file.exists():
                    with open(data_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                item = json.loads(line)
                                question = item.get('question', '')
                                options = [
                                    item.get('opa', ''),
                                    item.get('opb', ''),
                                    item.get('opc', ''),
                                    item.get('opd', '')
                                ]
                                answer_idx = item.get('cop', 0)
                                answer = self.ANSWER_MAP.get(answer_idx, 'A')
                                subject = item.get('subject_name', 'unknown')
                                
                                if question and any(options):
                                    samples.append(BenchmarkSample(
                                        id=f"medmcqa_{split}_{len(samples)}",
                                        dataset="medmcqa",
                                        text=question,
                                        options=options,
                                        label=answer,
                                        label_id=answer_idx if isinstance(answer_idx, int) else 0,
                                        difficulty=self._estimate_difficulty(question, options),
                                        subdomain=subject,
                                    ))
                            except:
                                continue
                    
                    if samples:
                        print(f"✓ Loaded {len(samples)} MedMCQA samples from JSON ({split})")
                    break
        
        if not samples:
            print(f"⚠ MedMCQA data not found in {self.data_dir}")
            return []
        
        if max_samples and len(samples) > max_samples:
            random.seed(seed)
            samples = random.sample(samples, max_samples)
            print(f"  Randomly selected {max_samples} samples (seed={seed})")
        
        self.samples = samples
        return samples
    
    def _estimate_difficulty(self, question: str, options: List[str]) -> str:
        """估计医学问题难度"""
        # 基于问题长度和选项复杂度估计
        q_words = len(question.split())
        avg_opt_len = sum(len(opt.split()) for opt in options) / 4
        
        if q_words > 30 or avg_opt_len > 10:
            return "hard"
        elif q_words < 15 and avg_opt_len < 5:
            return "easy"
        else:
            return "medium"
    
    def get_prompt_template(self) -> str:
        return """请回答以下医学多选题。

问题: {text}

选项:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

请选择最正确的答案，直接输出: answer: A/B/C/D"""


# ========== AlpacaEval 适配器 ==========
class AlpacaEvalAdapter(BaseAdapter):
    """
    AlpacaEval 开放式指令跟随数据集适配器
    
    数据集特点:
    - 805个开放式指令
    - 来源: helpful_base, koala, oasst, selfinstruct, vicuna
    - 评估方式: Pairwise comparison (与baseline对比的胜率)
    - 适合测试IARO的隐含意图理解能力
    """
    
    # 数据来源分类
    DATASET_SOURCES = ['helpful_base', 'koala', 'oasst', 'selfinstruct', 'vicuna']
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            project_root = Path(__file__).parent
            data_dir = project_root / "data" / "alpaca_eval" / "main"
        super().__init__(data_dir)
        self.difficulty_map = {}  # instruction index -> difficulty score
        self._load_difficulty_scores()
    
    def _load_difficulty_scores(self):
        """加载指令难度评分"""
        difficulty_file = self.data_dir / "instruction_difficulty.csv"
        if difficulty_file.exists():
            try:
                with open(difficulty_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        idx = int(row.get('index', 0))
                        score = float(row.get('instruction_difficulty', 0.5))
                        self.difficulty_map[idx] = score
            except Exception as e:
                print(f"⚠ Failed to load difficulty scores: {e}")
    
    def get_task_type(self) -> str:
        return "generation"  # 开放式生成任务
    
    def get_label_list(self) -> List[str]:
        return []  # 开放式任务没有固定标签
    
    def load(self, split: str = "eval", max_samples: int = None, seed: int = 42,
             source: str = None) -> List[BenchmarkSample]:
        """
        加载AlpacaEval数据集
        
        Args:
            split: 数据划分 (eval)
            max_samples: 最大样本数
            seed: 随机种子
            source: 筛选特定来源 (helpful_base, koala, oasst, selfinstruct, vicuna)
        """
        data_file = self.data_dir / "alpaca_eval.json"
        
        if not data_file.exists():
            print(f"⚠ AlpacaEval data file not found: {data_file}")
            return []
        
        samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for idx, item in enumerate(data):
            dataset_source = item.get('dataset', 'unknown')
            
            # 如果指定了来源，只加载该来源的数据
            if source and dataset_source != source:
                continue
            
            instruction = item.get('instruction', '')
            reference_output = item.get('output', '')  # text_davinci_003 的输出作为参考
            
            if not instruction:
                continue
            
            # 获取难度评分
            difficulty_score = self.difficulty_map.get(idx, 0.5)
            if difficulty_score < 0.3:
                difficulty = "easy"
            elif difficulty_score > 0.7:
                difficulty = "hard"
            else:
                difficulty = "medium"
            
            samples.append(BenchmarkSample(
                id=f"alpaca_eval_{idx}",
                dataset="alpaca_eval",
                text=instruction,
                context=reference_output,  # 将参考输出存在context中，用于评估
                label="",  # 开放式任务没有标准答案
                label_id=-1,
                difficulty=difficulty,
                subdomain=dataset_source,
                metadata={
                    "source": dataset_source,
                    "generator": item.get('generator', 'text_davinci_003'),
                    "difficulty_score": difficulty_score,
                    "index": idx
                }
            ))
        
        print(f"✓ Loaded {len(samples)} AlpacaEval samples")
        if source:
            print(f"  Filtered by source: {source}")
        
        # 显示来源分布
        source_dist = {}
        for s in samples:
            src = s.subdomain
            source_dist[src] = source_dist.get(src, 0) + 1
        print(f"  Source distribution: {source_dist}")
        
        # 随机采样
        if max_samples and len(samples) > max_samples:
            random.seed(seed)
            samples = random.sample(samples, max_samples)
            print(f"  Randomly selected {max_samples} samples (seed={seed})")
        
        self.samples = samples
        return samples
    
    def load_baseline_outputs(self, baseline: str = "gpt4") -> Dict[str, str]:
        """
        加载baseline模型的输出
        
        Args:
            baseline: "davinci" (text_davinci_003) 或 "gpt4"
        
        Returns:
            {instruction: output} 映射
        """
        if baseline == "gpt4":
            baseline_file = self.data_dir / "alpaca_eval_gpt4_baseline.json"
        else:
            baseline_file = self.data_dir / "alpaca_eval.json"
        
        if not baseline_file.exists():
            print(f"⚠ Baseline file not found: {baseline_file}")
            return {}
        
        outputs = {}
        with open(baseline_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            instruction = item.get('instruction', '')
            output = item.get('output', '')
            if instruction:
                outputs[instruction] = output
        
        print(f"✓ Loaded {len(outputs)} baseline outputs ({baseline})")
        return outputs
    
    def get_prompt_template(self) -> str:
        """AlpacaEval 的基础提示模板"""
        return """{instruction}"""
    
    def get_iaro_prompt_template(self) -> str:
        """IARO 增强的提示模板"""
        return """请回答以下用户指令。

## 用户指令
{instruction}

## 回答要求
1. 理解用户的核心需求和可能的隐含期望
2. 提供完整、准确、有帮助的回答
3. 注意回答的格式和风格要符合指令的要求

请直接给出回答:"""


# ========== 工厂函数 ==========
def get_adapter(dataset: str, data_dir: str = None) -> BaseAdapter:
    """获取数据集适配器"""
    adapters = {
        "sst5": SST5Adapter,
        "race": RACEAdapter,
        "medmcqa": MedMCQAAdapter,
        "alpaca_eval": AlpacaEvalAdapter,
    }
    
    if dataset.lower() not in adapters:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(adapters.keys())}")
    
    return adapters[dataset.lower()](data_dir)


def load_benchmark_data(
    dataset: str,
    split: str = "dev",
    max_samples: int = None,
    seed: int = 42,
    **kwargs
) -> Tuple[List[BenchmarkSample], BaseAdapter]:
    """
    便捷函数: 加载基准测试数据
    
    Args:
        dataset: 数据集名称 (sst5, race, medmcqa, alpaca_eval)
        split: 数据划分
        max_samples: 最大样本数
        seed: 随机种子
        **kwargs: 额外参数 (如RACE的level, AlpacaEval的source)
    
    Returns:
        (samples, adapter)
    """
    adapter = get_adapter(dataset)
    samples = adapter.load(split=split, max_samples=max_samples, seed=seed, **kwargs)
    return samples, adapter


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Benchmark Adapters")
    print("=" * 60)
    
    # 测试SST5
    print("\n[1] Testing SST5Adapter...")
    try:
        sst5_samples, _ = load_benchmark_data("sst5", split="dev", max_samples=5)
        if sst5_samples:
            print(f"  Sample: {sst5_samples[0].text[:50]}...")
            print(f"  Label: {sst5_samples[0].label}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 测试RACE
    print("\n[2] Testing RACEAdapter...")
    try:
        race_samples, _ = load_benchmark_data("race", split="dev", max_samples=5, level="high")
        if race_samples:
            print(f"  Question: {race_samples[0].text[:50]}...")
            print(f"  Answer: {race_samples[0].label}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 测试MedMCQA
    print("\n[3] Testing MedMCQAAdapter...")
    try:
        med_samples, _ = load_benchmark_data("medmcqa", split="dev", max_samples=5)
        if med_samples:
            print(f"  Question: {med_samples[0].text[:50]}...")
            print(f"  Answer: {med_samples[0].label}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 测试AlpacaEval
    print("\n[4] Testing AlpacaEvalAdapter...")
    try:
        alpaca_samples, adapter = load_benchmark_data("alpaca_eval", split="eval", max_samples=5)
        if alpaca_samples:
            print(f"  Instruction: {alpaca_samples[0].text[:80]}...")
            print(f"  Source: {alpaca_samples[0].subdomain}")
            print(f"  Difficulty: {alpaca_samples[0].difficulty}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Adapter testing completed")
    print("=" * 60)
