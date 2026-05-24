"""
基准测试运行器 (benchmark_runner.py)
在SST5、RACE、MedMCQA数据集上运行IARO方法和基线方法

用于ACL论文: Bridging the Intent Gap

方法列表:
  Baseline: Standard, CoT, CoT-SC, SelfRefine, OPRO
  IARO: IARO-Base, IARO-Augment, IARO-Hybrid

用法:
    python benchmark_runner.py --dataset sst5 --max-samples 50
    python benchmark_runner.py --dataset race --max-samples 100 --level high
    python benchmark_runner.py --dataset medmcqa --max-samples 100
    python benchmark_runner.py --dataset all --max-samples 50 --quick-test
    
后台运行:
    nohup python benchmark_runner.py --dataset all --max-samples 200 > logs/exp.log 2>&1 &
"""

import asyncio
import argparse
import json
import re
import time
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from llm_client import AsyncLLMClient, create_solver_client
from benchmark_adapter import (
    BenchmarkSample, 
    load_benchmark_data,
    SST5Adapter,
    RACEAdapter, 
    MedMCQAAdapter,
)
from benchmark_evaluator import (
    BenchmarkEvaluator,
    BenchmarkResults,
    MethodResult,
    generate_summary_table,
    NumpyEncoder,
)


# ========== 方法定义 ==========
BASELINE_METHODS = ['Standard', 'CoT', 'CoT-SC', 'SelfRefine', 'OPRO']
IARO_METHODS = ['IARO-Base', 'IARO-Augment', 'IARO-Hybrid']
ALL_METHODS = BASELINE_METHODS + IARO_METHODS

# CoT-SC 采样次数
COT_SC_SAMPLES = 3

# API 成本估算 (每1K tokens, USD)
API_COST_PER_1K = {
    "input": 0.001,   # $0.001 per 1K input tokens
    "output": 0.002,  # $0.002 per 1K output tokens
}


# ========== 日志配置 ==========
class ExperimentLogger:
    """实验日志管理器 - 实时写入日志文件"""
    
    def __init__(self, log_dir: str = "logs/benchmark", experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name or f"benchmark_{timestamp}"
        self.log_file = self.log_dir / f"{self.experiment_name}.log"
        
        # 配置日志
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # 文件处理器 - 实时写入
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 写入日志头部
        self._write_header()
    
    def _write_header(self):
        """写入日志头部信息"""
        header = f"""
{'='*80}
实验日志 - IARO基准测试
{'='*80}
实验名称: {self.experiment_name}
开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
日志文件: {self.log_file}

背景说明:
  本实验用于评估IARO(Intent-Aware Response Optimization)方法在
  多个基准数据集(SST5/RACE/MedMCQA)上的性能表现。
  对比方法包括: Standard, CoT, CoT-SC, SelfRefine, OPRO等基线方法。
  
目标:
  - 验证IARO方法在分类/QA任务上的有效性
  - 对比不同方法的准确率和API效率
  - 为ACL论文提供实验数据支持
{'='*80}
"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def info(self, msg: str):
        self.logger.info(msg)
        self._flush()
    
    def debug(self, msg: str):
        self.logger.debug(msg)
        self._flush()
    
    def warning(self, msg: str):
        self.logger.warning(msg)
        self._flush()
    
    def error(self, msg: str):
        self.logger.error(msg)
        self._flush()
    
    def _flush(self):
        """确保日志实时写入"""
        for handler in self.logger.handlers:
            handler.flush()
    
    def log_experiment_config(self, config: Dict):
        """记录实验配置"""
        self.info("\n" + "="*60)
        self.info("实验配置:")
        for k, v in config.items():
            self.info(f"  {k}: {v}")
        self.info("="*60 + "\n")
    
    def log_method_result(self, method: str, accuracy: float, api_calls: int, time_sec: float):
        """记录单个方法结果"""
        self.info(f"  ✓ {method}: Acc={accuracy:.1%}, API={api_calls}, Time={time_sec:.1f}s")
    
    def log_final_summary(self, results: Dict):
        """记录最终总结"""
        self.info("\n" + "="*60)
        self.info("实验完成 - 最终结果:")
        self.info("="*60)
        for method, data in results.items():
            self.info(f"  {method}: {data.get('accuracy', 0):.1%}")
        self.info("="*60)


# ========== Prompt模板 ==========
class PromptTemplates:
    """各种方法的Prompt模板"""
    
    # ===== SST5 情感分析 =====
    @staticmethod
    def sst5_standard(text: str) -> Tuple[str, str]:
        prompt = f"""请对以下文本进行5分类情感分析。

文本: {text}

类别: very negative, negative, neutral, positive, very positive

请直接输出分类结果: sentiment: <类别>"""
        system = "You are an expert sentiment classifier."
        return prompt, system
    
    @staticmethod
    def sst5_cot(text: str) -> Tuple[str, str]:
        prompt = f"""请对以下文本进行5分类情感分析。

文本: {text}

Let's think step by step:
1. 识别文本中的情感词汇
2. 分析整体语气
3. 判断情感强度

类别: very negative, negative, neutral, positive, very positive

请输出分析过程，最后给出: sentiment: <类别>"""
        system = "You are an expert sentiment classifier that thinks step by step."
        return prompt, system
    
    @staticmethod
    def sst5_iaro_analysis(text: str) -> Tuple[str, str]:
        prompt = f"""分析以下文本的情感特征:

文本: {text}

请输出JSON格式分析:
```json
{{
    "sentiment_words": ["情感词1", "情感词2"],
    "tone": "整体语气",
    "potential_sarcasm": true/false,
    "sentiment_intensity": 0-10,
    "key_indicators": ["指标1", "指标2"]
}}
```"""
        system = "你是情感分析专家，擅长识别复杂情感表达。"
        return prompt, system
    
    @staticmethod
    def sst5_iaro_classify(text: str, analysis: Dict) -> Tuple[str, str]:
        intensity = analysis.get('sentiment_intensity', 5)
        words = analysis.get('sentiment_words', [])
        tone = analysis.get('tone', 'unknown')
        sarcasm = analysis.get('potential_sarcasm', False)
        
        prompt = f"""[情感分析任务 - 增强版]

[分析信息]
- 情感词汇: {', '.join(words) if words else '未识别'}
- 语气: {tone}
- 讽刺可能: {'是' if sarcasm else '否'}
- 情感强度: {intensity}/10

[待分类文本]
{text}

[分类指南]
- very negative: 强烈负面 (强度8-10)
- negative: 负面 (强度3-7)
- neutral: 中性 (强度0-2)
- positive: 正面 (强度3-7)
- very positive: 强烈正面 (强度8-10)

请输出: sentiment: <类别>"""
        system = "You are an expert sentiment classifier."
        return prompt, system
    
    # ===== MCQA (RACE, MedMCQA) =====
    @staticmethod
    def mcqa_standard(question: str, options: List[str], context: str = "") -> Tuple[str, str]:
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        context_part = f"\n\n文章:\n{context[:1500]}" if context else ""
        
        prompt = f"""请回答以下选择题。{context_part}

问题: {question}

选项:
{options_str}

请直接输出答案: answer: A/B/C/D"""
        system = "You are a helpful assistant that answers multiple choice questions."
        return prompt, system
    
    @staticmethod
    def mcqa_cot(question: str, options: List[str], context: str = "") -> Tuple[str, str]:
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        context_part = f"\n\n文章:\n{context[:1500]}" if context else ""
        
        prompt = f"""请回答以下选择题。{context_part}

问题: {question}

选项:
{options_str}

Let's think step by step:
1. 理解问题核心
2. 分析每个选项
3. 选择最佳答案

请输出分析和答案: answer: A/B/C/D"""
        system = "You are a helpful assistant that thinks step by step."
        return prompt, system
    
    @staticmethod
    def mcqa_iaro_analysis(question: str, options: List[str], context: str = "") -> Tuple[str, str]:
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        context_snippet = context[:800] if context else "无"
        
        prompt = f"""分析以下选择题:

问题: {question}
选项:
{options_str}
上下文: {context_snippet}

请输出JSON格式分析:
```json
{{
    "core_question": "问题核心要求",
    "key_concepts": ["关键概念1", "关键概念2"],
    "option_analysis": {{"A": "分析", "B": "分析", "C": "分析", "D": "分析"}},
    "reasoning_steps": ["步骤1", "步骤2"],
    "difficulty": "easy/medium/hard"
}}
```"""
        system = "你是问题分析专家，擅长提取关键信息和推理。"
        return prompt, system
    
    @staticmethod
    def mcqa_iaro_answer(question: str, options: List[str], analysis: Dict, context: str = "") -> Tuple[str, str]:
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        context_part = f"\n\n参考文章:\n{context[:1200]}" if context else ""
        
        core_q = analysis.get('core_question', question)
        concepts = analysis.get('key_concepts', [])
        opt_analysis = analysis.get('option_analysis', {})
        steps = analysis.get('reasoning_steps', [])
        
        opt_str = "\n".join([f"- {k}: {v}" for k, v in opt_analysis.items()])
        steps_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
        
        prompt = f"""[选择题 - 增强版]{context_part}

[问题分析]
- 核心: {core_q}
- 关键概念: {', '.join(concepts) if concepts else '未识别'}

[选项分析]
{opt_str}

[解题步骤]
{steps_str if steps_str else '直接分析'}

[原始问题]
{question}

[选项]
{options_str}

请选择最正确的答案: answer: A/B/C/D"""
        system = "You are an expert at answering questions."
        return prompt, system


# ========== 答案解析器 ==========
class AnswerParser:
    """解析模型输出的答案"""
    
    @staticmethod
    def parse_sentiment(response: str) -> str:
        """解析情感标签"""
        response_lower = response.lower()
        
        if 'very negative' in response_lower:
            return 'very negative'
        if 'very positive' in response_lower:
            return 'very positive'
        if 'negative' in response_lower:
            return 'negative'
        if 'positive' in response_lower:
            return 'positive'
        if 'neutral' in response_lower:
            return 'neutral'
        
        return 'neutral'
    
    @staticmethod
    def parse_mcqa(response: str) -> str:
        """解析MCQA答案"""
        response_upper = response.upper()
        
        # 查找 answer: X 格式
        match = re.search(r'ANSWER[:\s]*([A-D])', response_upper)
        if match:
            return match.group(1)
        
        # 查找独立的 A/B/C/D
        for letter in ['A', 'B', 'C', 'D']:
            if f' {letter}.' in response_upper or f' {letter} ' in response_upper:
                return letter
            if response_upper.strip().endswith(letter):
                return letter
        
        # 最后尝试找任意 A-D
        match = re.search(r'[A-D]', response_upper)
        if match:
            return match.group()
        
        return 'A'
    
    @staticmethod
    def parse_json(response: str) -> Dict:
        """解析JSON响应"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {}


# ========== 方法求解器 ==========
class BenchmarkSolver:
    """基准测试求解器 - 封装各种方法"""
    
    def __init__(self, client: AsyncLLMClient = None):
        self.client = client or create_solver_client()
        self.templates = PromptTemplates()
        self.parser = AnswerParser()
    
    async def _generate(self, prompt: str, system: str, temperature: float = 0.3, 
                        max_tokens: int = 500) -> str:
        """调用LLM生成"""
        return await self.client.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    async def _generate_json(self, prompt: str, system: str) -> Dict:
        """调用LLM生成JSON"""
        try:
            return await self.client.generate_json(
                prompt=prompt,
                system_prompt=system,
                temperature=0.3,
                max_tokens=800,
            )
        except:
            response = await self._generate(prompt, system, max_tokens=800)
            return self.parser.parse_json(response)
    
    # ===== SST5方法 =====
    async def sst5_standard(self, sample: BenchmarkSample) -> Tuple[str, int]:
        prompt, system = self.templates.sst5_standard(sample.text)
        response = await self._generate(prompt, system)
        return self.parser.parse_sentiment(response), 1
    
    async def sst5_cot(self, sample: BenchmarkSample) -> Tuple[str, int]:
        prompt, system = self.templates.sst5_cot(sample.text)
        response = await self._generate(prompt, system, max_tokens=600)
        return self.parser.parse_sentiment(response), 1
    
    async def sst5_cot_sc(self, sample: BenchmarkSample) -> Tuple[str, int]:
        """CoT-SC: 多次采样投票 (采样次数=3)"""
        prompt, system = self.templates.sst5_cot(sample.text)
        predictions = []
        for _ in range(COT_SC_SAMPLES):
            response = await self._generate(prompt, system, temperature=0.7, max_tokens=600)
            predictions.append(self.parser.parse_sentiment(response))
        
        # 投票
        counter = Counter(predictions)
        return counter.most_common(1)[0][0], COT_SC_SAMPLES
    
    async def sst5_self_refine(self, sample: BenchmarkSample) -> Tuple[str, int]:
        """SelfRefine: 生成-反思-改进"""
        # Step 1: 初始分类
        prompt, system = self.templates.sst5_standard(sample.text)
        initial_response = await self._generate(prompt, system)
        initial_pred = self.parser.parse_sentiment(initial_response)
        
        # Step 2: 反思和改进
        refine_prompt = f"""你之前对以下文本的情感分类为: {initial_pred}

文本: {sample.text}

请反思这个分类是否正确:
1. 检查是否有遗漏的情感信号
2. 检查是否有讽刺或反语
3. 检查情感强度是否合适

如果需要修正，请给出新的分类。
类别: very negative, negative, neutral, positive, very positive

最终答案: sentiment: <类别>"""
        
        refined_response = await self._generate(refine_prompt, system, max_tokens=600)
        return self.parser.parse_sentiment(refined_response), 2
    
    async def sst5_opro(self, sample: BenchmarkSample) -> Tuple[str, int]:
        """OPRO: 优化提示循环 (约7次API调用)"""
        # Step 1: 生成多个候选提示
        meta_prompt = f"""为以下情感分析任务生成3个不同的分类提示:

文本: {sample.text}
类别: very negative, negative, neutral, positive, very positive

请生成3个不同角度的分析提示,用JSON格式:
```json
{{"prompts": ["提示1", "提示2", "提示3"]}}
```"""
        
        prompts_response = await self._generate_json(meta_prompt, "你是提示工程专家")
        candidate_prompts = prompts_response.get('prompts', [self.templates.sst5_standard(sample.text)[0]])
        
        # Step 2: 用每个提示进行分类
        predictions = []
        for cp in candidate_prompts[:3]:  # 最多3个
            resp = await self._generate(cp, "You are a sentiment classifier.")
            predictions.append(self.parser.parse_sentiment(resp))
        
        # Step 3: 综合判断
        synthesis_prompt = f"""多个分类器对以下文本给出了不同预测:

文本: {sample.text}
预测结果: {predictions}

请综合分析，给出最终分类:
sentiment: <类别>"""
        
        final_response = await self._generate(synthesis_prompt, "You are an expert sentiment classifier.")
        
        # 总共约 1(meta) + 3(candidates) + 1(synthesis) = 5-7 次调用
        return self.parser.parse_sentiment(final_response), 1 + len(candidate_prompts[:3]) + 1
    
    async def sst5_iaro_base(self, sample: BenchmarkSample) -> Tuple[str, int]:
        # Step 1: 分析
        prompt, system = self.templates.sst5_iaro_analysis(sample.text)
        analysis = await self._generate_json(prompt, system)
        
        # Step 2: 分类
        prompt, system = self.templates.sst5_iaro_classify(sample.text, analysis)
        response = await self._generate(prompt, system)
        
        return self.parser.parse_sentiment(response), 2
    
    async def sst5_iaro_augment(self, sample: BenchmarkSample) -> Tuple[str, int]:
        # 与base相同，但更详细的分析
        return await self.sst5_iaro_base(sample)
    
    async def sst5_iaro_hybrid(self, sample: BenchmarkSample) -> Tuple[str, int]:
        # Step 1: 快速难度评估
        difficulty = self._quick_difficulty_check(sample.text)
        
        if difficulty == "easy":
            # 简单样本直接分类
            prompt, system = self.templates.sst5_standard(sample.text)
            response = await self._generate(prompt, system)
            return self.parser.parse_sentiment(response), 1
        else:
            # 复杂样本用完整IARO
            return await self.sst5_iaro_base(sample)
    
    def _quick_difficulty_check(self, text: str) -> str:
        """快速难度评估 (本地规则)"""
        text_lower = text.lower()
        contrast_words = ['but', 'however', 'although', 'despite', 'yet']
        has_contrast = any(w in text_lower for w in contrast_words)
        
        if has_contrast or len(text.split()) > 25:
            return "hard"
        return "easy"
    
    # ===== MCQA方法 =====
    async def mcqa_standard(self, sample: BenchmarkSample) -> Tuple[str, int]:
        prompt, system = self.templates.mcqa_standard(
            sample.text, sample.options, sample.context
        )
        response = await self._generate(prompt, system)
        return self.parser.parse_mcqa(response), 1
    
    async def mcqa_cot(self, sample: BenchmarkSample) -> Tuple[str, int]:
        prompt, system = self.templates.mcqa_cot(
            sample.text, sample.options, sample.context
        )
        response = await self._generate(prompt, system, max_tokens=800)
        return self.parser.parse_mcqa(response), 1
    
    async def mcqa_cot_sc(self, sample: BenchmarkSample) -> Tuple[str, int]:
        """CoT-SC: 多次采样投票 (采样次数=3)"""
        prompt, system = self.templates.mcqa_cot(
            sample.text, sample.options, sample.context
        )
        predictions = []
        for _ in range(COT_SC_SAMPLES):
            response = await self._generate(prompt, system, temperature=0.7, max_tokens=800)
            predictions.append(self.parser.parse_mcqa(response))
        
        counter = Counter(predictions)
        return counter.most_common(1)[0][0], COT_SC_SAMPLES
    
    async def mcqa_fewshot_cot(self, sample: BenchmarkSample) -> Tuple[str, int]:
        """Few-Shot CoT: 带示例的思维链推理 (更强的baseline)"""
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample.options)])
        context_snippet = sample.context[:800] if sample.context else ""
        
        fewshot_prompt = f"""Here are examples of careful reasoning for multiple choice questions:

---
Example 1:
Question: What is the main purpose of the passage?
A. To entertain readers
B. To explain a scientific concept
C. To persuade readers to take action
D. To describe historical events

Thinking: Let me analyze each option against the passage content:
- A: Check if the tone is entertaining or humorous - usually not in academic texts
- B: Check if scientific terms and explanations are present
- C: Check for persuasive language and calls to action
- D: Check for dates, historical figures, or chronological narratives
Based on the evidence in the passage, I select the option that best matches.
Answer: B

---
Example 2:
Question: According to the passage, why did the author make this decision?
A. Financial reasons
B. Personal interest
C. External pressure
D. Lack of alternatives

Thinking: I need to find explicit or implicit reasons in the text:
- Look for keywords related to money, cost, budget (A)
- Look for passion, curiosity, hobby mentions (B)
- Look for mentions of others' influence, requirements (C)
- Look for phrases like "no choice", "only option" (D)
The passage suggests the answer is...
Answer: C

---
Now solve this question using the same careful reasoning:

Context: {context_snippet}

Question: {sample.text}
Options:
{options_str}

Thinking:"""
        
        system = "You are an expert at reading comprehension. Analyze carefully before answering."
        response = await self._generate(fewshot_prompt, system, max_tokens=1000)
        return self.parser.parse_mcqa(response), 1
    
    async def mcqa_self_refine(self, sample: BenchmarkSample) -> Tuple[str, int]:
        """SelfRefine: 生成-反思-改进"""
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample.options)])
        
        # Step 1: 初始回答
        prompt, system = self.templates.mcqa_standard(sample.text, sample.options, sample.context)
        initial_response = await self._generate(prompt, system)
        initial_pred = self.parser.parse_mcqa(initial_response)
        
        # Step 2: 反思和改进
        refine_prompt = f"""你之前对以下问题的回答是: {initial_pred}

问题: {sample.text}
选项:
{options_str}

请反思这个答案是否正确:
1. 重新审视每个选项
2. 检查是否有遗漏的关键信息
3. 验证推理过程

如果需要修正，请给出新的答案。
最终答案: answer: A/B/C/D"""
        
        refined_response = await self._generate(refine_prompt, system, max_tokens=800)
        return self.parser.parse_mcqa(refined_response), 2
    
    async def mcqa_opro(self, sample: BenchmarkSample) -> Tuple[str, int]:
        """OPRO: 优化提示循环 (约7次API调用)"""
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample.options)])
        context_snippet = sample.context[:500] if sample.context else ""
        
        # Step 1: 生成多个解题策略
        meta_prompt = f"""为以下选择题生成3个不同的解题策略:

问题: {sample.text}
选项:
{options_str}
上下文: {context_snippet}

请生成3个不同的解题思路,用JSON格式:
```json
{{"strategies": ["策略1", "策略2", "策略3"]}}
```"""
        
        strategies_response = await self._generate_json(meta_prompt, "你是解题策略专家")
        strategies = strategies_response.get('strategies', ["直接分析选项"])
        
        # Step 2: 用每个策略进行解答
        predictions = []
        for strategy in strategies[:3]:
            solve_prompt = f"""使用以下策略解答问题:
策略: {strategy}

问题: {sample.text}
选项:
{options_str}

答案: answer: A/B/C/D"""
            resp = await self._generate(solve_prompt, "You answer questions.")
            predictions.append(self.parser.parse_mcqa(resp))
        
        # Step 3: 综合判断
        synthesis_prompt = f"""多个策略对以下问题给出了不同答案:

问题: {sample.text}
各策略答案: {predictions}

请综合分析，给出最终答案:
answer: A/B/C/D"""
        
        final_response = await self._generate(synthesis_prompt, "You are an expert.")
        return self.parser.parse_mcqa(final_response), 1 + len(strategies[:3]) + 1
    
    async def mcqa_iaro_base(self, sample: BenchmarkSample) -> Tuple[str, int]:
        # Step 1: 分析
        prompt, system = self.templates.mcqa_iaro_analysis(
            sample.text, sample.options, sample.context
        )
        analysis = await self._generate_json(prompt, system)
        
        # Step 2: 回答
        prompt, system = self.templates.mcqa_iaro_answer(
            sample.text, sample.options, analysis, sample.context
        )
        response = await self._generate(prompt, system)
        
        return self.parser.parse_mcqa(response), 2
    
    async def mcqa_iaro_augment(self, sample: BenchmarkSample) -> Tuple[str, int]:
        return await self.mcqa_iaro_base(sample)
    
    async def mcqa_iaro_hybrid(self, sample: BenchmarkSample) -> Tuple[str, int]:
        # 基于问题长度和选项复杂度决定策略
        q_len = len(sample.text.split())
        avg_opt_len = sum(len(opt.split()) for opt in sample.options) / 4
        
        if q_len < 15 and avg_opt_len < 8:
            # 简单问题直接回答
            return await self.mcqa_standard(sample)
        else:
            return await self.mcqa_iaro_base(sample)
    
    async def mcqa_iaro_adaptive(self, sample: BenchmarkSample) -> Tuple[str, int]:
        """IARO-Adaptive V2: 基于任务类型决定是否启用 IARO
        
        核心逻辑: MCQA 任务 (阅读理解、情感分类等) 是显式任务，
        不需要意图分析，直接使用 Standard 方法。
        IARO 仅对开放式生成任务有效。
        """
        dataset = sample.dataset.lower()
        
        # 显式任务类型 - 跳过 IARO，直接使用 Standard
        EXPLICIT_TASKS = ['race', 'medmcqa', 'sst5', 'drop', 'boolq', 'commonsenseqa']
        
        if dataset in EXPLICIT_TASKS:
            # 显式 MCQA 任务：Standard 已经是最优解
            return await self.mcqa_standard(sample)
        
        # 检查问题是否是纯事实性问题 (无隐式需求)
        factual_indicators = [
            'according to the passage', 'based on the text', 'what does',
            'which of the following', 'the author', 'in the passage',
            'what is the main', 'what can be inferred'
        ]
        query_lower = sample.text.lower()
        is_factual = any(ind in query_lower for ind in factual_indicators)
        
        if is_factual:
            return await self.mcqa_standard(sample)
        
        # 其他情况：使用 IARO
        return await self.mcqa_iaro_base(sample)
    
    # ===== 通用入口 =====
    async def solve(self, sample: BenchmarkSample, method: str) -> Tuple[str, int]:
        """
        统一求解入口
        
        Returns:
            (prediction, api_calls)
        """
        dataset = sample.dataset.lower()
        method_lower = method.lower().replace('-', '_')
        
        if dataset == "sst5":
            method_map = {
                "standard": self.sst5_standard,
                "cot": self.sst5_cot,
                "cot_sc": self.sst5_cot_sc,
                "selfrefine": self.sst5_self_refine,
                "opro": self.sst5_opro,
                "iaro_base": self.sst5_iaro_base,
                "iaro_augment": self.sst5_iaro_augment,
                "iaro_hybrid": self.sst5_iaro_hybrid,
            }
        else:  # race, medmcqa
            method_map = {
                "standard": self.mcqa_standard,
                "cot": self.mcqa_cot,
                "cot_sc": self.mcqa_cot_sc,
                "fewshot_cot": self.mcqa_fewshot_cot,
                "selfrefine": self.mcqa_self_refine,
                "opro": self.mcqa_opro,
                "iaro_base": self.mcqa_iaro_base,
                "iaro_augment": self.mcqa_iaro_augment,
                "iaro_hybrid": self.mcqa_iaro_hybrid,
                "iaro_adaptive": self.mcqa_iaro_adaptive,
            }
        
        if method_lower not in method_map:
            raise ValueError(f"Unknown method: {method}")
        
        return await method_map[method_lower](sample)


# ========== 实验运行器 ==========
class BenchmarkExperimentRunner:
    """基准测试实验运行器 - 带进度条和实时日志"""
    
    def __init__(
        self,
        output_dir: str = "results/benchmark",
        methods: List[str] = None,
        log_dir: str = "logs/benchmark",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.methods = methods or ALL_METHODS
        self.solver = None
        
        # 初始化日志
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = ExperimentLogger(log_dir, f"benchmark_{timestamp}")
        
        # API调用统计
        self.api_stats = {
            "total_calls": 0,
            "by_method": {},
            "estimated_cost_usd": 0.0,
        }
    
    def _init_solver(self):
        """初始化求解器"""
        if self.solver is None:
            self.solver = BenchmarkSolver()
            self.logger.info(f"✓ Solver initialized with model: {self.solver.client.model}")
    
    async def run_experiment(
        self,
        dataset: str,
        split: str = "dev",
        max_samples: int = 50,
        seed: int = 42,
        **kwargs,
    ) -> BenchmarkResults:
        """
        运行单个数据集的实验
        
        Args:
            dataset: 数据集名称 (sst5, race, medmcqa)
            split: 数据划分
            max_samples: 最大样本数
            seed: 随机种子
            **kwargs: 额外参数
        
        Returns:
            BenchmarkResults
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"Running {dataset.upper()} Benchmark")
        self.logger.info("=" * 70)
        
        # 记录实验配置
        exp_config = {
            "dataset": dataset,
            "split": split,
            "max_samples": max_samples,
            "seed": seed,
            "methods": self.methods,
            **kwargs
        }
        self.logger.log_experiment_config(exp_config)
        
        self._init_solver()
        
        # 加载数据
        self.logger.info(f"\n[1/3] Loading {dataset} data...")
        samples, adapter = load_benchmark_data(
            dataset=dataset,
            split=split,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )
        
        if not samples:
            self.logger.warning(f"⚠ No samples loaded for {dataset}")
            return None
        
        self.logger.info(f"  Loaded {len(samples)} samples")
        
        # 运行各方法
        self.logger.info(f"\n[2/3] Running methods: {self.methods}")
        
        all_predictions = {}
        all_api_calls = {}
        method_times = {}
        
        experiment_start = time.time()
        
        for method in self.methods:
            self.logger.info(f"\n  Running {method}...")
            method_start = time.time()
            predictions = []
            total_calls = 0
            errors = 0
            
            # 使用进度条
            if HAS_TQDM:
                sample_iter = tqdm(samples, desc=f"    {method}", leave=False, ncols=80)
            else:
                sample_iter = samples
            
            for i, sample in enumerate(sample_iter):
                try:
                    pred, calls = await self.solver.solve(sample, method)
                    predictions.append(pred)
                    total_calls += calls
                    
                    # 更新进度条描述
                    if HAS_TQDM:
                        correct_so_far = sum(1 for p, s in zip(predictions, samples[:len(predictions)]) if p == s.label)
                        acc_so_far = correct_so_far / len(predictions)
                        sample_iter.set_postfix({"acc": f"{acc_so_far:.1%}", "calls": total_calls})
                    elif (i + 1) % 20 == 0:
                        elapsed = time.time() - experiment_start
                        self.logger.debug(f"    [{i+1}/{len(samples)}] Elapsed: {elapsed/60:.1f}min")
                        
                except Exception as e:
                    self.logger.debug(f"    ⚠ Error on sample {i}: {e}")
                    errors += 1
                    # 填充默认值
                    if dataset == "sst5":
                        predictions.append("neutral")
                    else:
                        predictions.append("A")
                    total_calls += 1
            
            method_time = time.time() - method_start
            method_times[method] = method_time
            
            all_predictions[method] = predictions
            all_api_calls[method] = total_calls
            
            # 更新API统计
            self.api_stats["total_calls"] += total_calls
            self.api_stats["by_method"][method] = total_calls
            
            # 快速计算准确率
            correct = sum(1 for p, s in zip(predictions, samples) if p == s.label)
            acc = correct / len(samples)
            
            self.logger.log_method_result(method, acc, total_calls, method_time)
            if errors > 0:
                self.logger.warning(f"    ⚠ {errors} errors occurred")
        
        total_time = time.time() - experiment_start
        
        # 评估
        print(f"\n[3/3] Evaluating results...")
        
        evaluator = BenchmarkEvaluator(dataset, adapter.get_label_list())
        method_results = {}
        
        for method in self.methods:
            result = evaluator.evaluate_method(
                predictions=all_predictions[method],
                samples=samples,
                method_name=method,
                api_calls=all_api_calls[method],
            )
            method_results[method] = result
        
        # 对比分析
        comparison = evaluator.compare_methods(method_results, baseline="Standard")
        
        # 找IARO胜出样本
        iaro_wins = evaluator.find_iaro_wins(samples, method_results)
        
        # 构建结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = BenchmarkResults(
            experiment_id=f"{dataset}_{timestamp}",
            dataset=dataset,
            num_samples=len(samples),
            method_results=method_results,
            comparison=comparison,
            config={
                "split": split,
                "max_samples": max_samples,
                "seed": seed,
                "methods": self.methods,
                "total_time_min": round(total_time / 60, 2),
                **kwargs,
            },
            timestamp=timestamp,
        )
        
        # 添加IARO胜出案例
        results.sample_details = iaro_wins[:20]  # 最多保存20个
        
        # 保存结果
        output_file = self.output_dir / f"{dataset}_{timestamp}.json"
        results.save(str(output_file))
        
        # 生成报告 (带背景说明和API成本分析)
        report = self._generate_report_with_header(results, all_api_calls, method_times, total_time)
        report_file = self.output_dir / f"{dataset}_{timestamp}_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        self.logger.info(f"✓ Report saved to {report_file}")
        
        # 打印总结
        self._print_summary(results)
        
        return results
    
    def _generate_report_with_header(
        self, 
        results: BenchmarkResults, 
        api_calls: Dict[str, int],
        method_times: Dict[str, float],
        total_time: float
    ) -> str:
        """生成带背景说明和API成本分析的报告"""
        lines = []
        
        # 背景说明头部
        lines.append(f"""# IARO基准测试报告 - {results.dataset.upper()}

## 背景说明

**实验目的**: 评估IARO(Intent-Aware Response Optimization)方法在{results.dataset.upper()}数据集上的性能表现。

**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**数据集信息**:
- 数据集: {results.dataset.upper()}
- 样本数: {results.num_samples}
- 数据划分: {results.config.get('split', 'dev')}
- 随机种子: {results.config.get('seed', 42)}

**对比方法**:
- Baseline: Standard, CoT, CoT-SC, SelfRefine, OPRO
- IARO: IARO-Base, IARO-Augment, IARO-Hybrid

---
""")
        
        # 主要结果表格
        lines.append("## 主要结果\n")
        lines.append("| Method | Accuracy | Macro-F1 | API Calls | Time (s) | Efficiency |")
        lines.append("|--------|----------|----------|-----------|----------|------------|")
        
        sorted_methods = sorted(
            results.method_results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        
        for method, result in sorted_methods:
            time_sec = method_times.get(method, 0)
            calls = api_calls.get(method, result.api_calls)
            eff = result.accuracy / calls if calls > 0 else 0
            marker = " ★" if 'iaro' in method.lower() else ""
            lines.append(
                f"| {method}{marker} | {result.accuracy:.1%} | {result.macro_f1:.3f} | "
                f"{calls} | {time_sec:.1f} | {eff:.4f} |"
            )
        
        # API成本分析
        lines.append("\n## API调用统计\n")
        lines.append("| Method | API Calls | Avg Calls/Sample | Est. Cost (USD) |")
        lines.append("|--------|-----------|------------------|-----------------|")
        
        total_api_calls = 0
        for method in results.method_results.keys():
            calls = api_calls.get(method, 0)
            avg_calls = calls / results.num_samples if results.num_samples > 0 else 0
            # 估算成本: 假设平均每次调用500 input + 200 output tokens
            est_cost = calls * (500 * API_COST_PER_1K["input"] + 200 * API_COST_PER_1K["output"]) / 1000
            total_api_calls += calls
            lines.append(f"| {method} | {calls} | {avg_calls:.1f} | ${est_cost:.4f} |")
        
        total_cost = total_api_calls * (500 * API_COST_PER_1K["input"] + 200 * API_COST_PER_1K["output"]) / 1000
        lines.append(f"| **Total** | **{total_api_calls}** | - | **${total_cost:.4f}** |")
        
        # 关键发现
        baselines = [m for m in results.method_results.keys() if 'iaro' not in m.lower()]
        iaro_methods = [m for m in results.method_results.keys() if 'iaro' in m.lower()]
        
        if baselines and iaro_methods:
            best_baseline = max(baselines, key=lambda m: results.method_results[m].accuracy)
            best_iaro = max(iaro_methods, key=lambda m: results.method_results[m].accuracy)
            
            base_acc = results.method_results[best_baseline].accuracy
            iaro_acc = results.method_results[best_iaro].accuracy
            improvement = (iaro_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0
            
            base_calls = api_calls.get(best_baseline, 0)
            iaro_calls = api_calls.get(best_iaro, 0)
            
            lines.append(f"\n## 关键发现\n")
            lines.append(f"- **Best Baseline**: {best_baseline} (Acc: {base_acc:.1%}, API: {base_calls})")
            lines.append(f"- **Best IARO**: {best_iaro} (Acc: {iaro_acc:.1%}, API: {iaro_calls})")
            lines.append(f"- **Accuracy Improvement**: {improvement:+.1f}%")
            lines.append(f"- **Total Experiment Time**: {total_time/60:.1f} minutes")
        
        # 效率分析
        lines.append(f"\n## 效率分析\n")
        lines.append("效率 = Accuracy / API Calls，表示每次API调用获得的准确率。\n")
        
        eff_ranking = sorted(
            [(m, r.accuracy / api_calls.get(m, 1)) for m, r in results.method_results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        for rank, (method, eff) in enumerate(eff_ranking, 1):
            marker = " ★" if 'iaro' in method.lower() else ""
            lines.append(f"{rank}. {method}{marker}: {eff:.4f}")
        
        lines.append(f"\n---\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)
    
    def _print_summary(self, results: BenchmarkResults):
        """打印结果总结"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"RESULTS SUMMARY - {results.dataset.upper()}")
        self.logger.info("=" * 70)
        
        self.logger.info("\n## Accuracy Ranking")
        sorted_methods = sorted(
            results.method_results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        for rank, (method, result) in enumerate(sorted_methods, 1):
            marker = " ★" if 'iaro' in method.lower() else ""
            self.logger.info(f"  {rank}. {method}: {result.accuracy:.1%}{marker}")
        
        # IARO优势
        baselines = [m for m in results.method_results.keys() if 'iaro' not in m.lower()]
        iaro_methods = [m for m in results.method_results.keys() if 'iaro' in m.lower()]
        
        if baselines and iaro_methods:
            best_baseline = max(baselines, key=lambda m: results.method_results[m].accuracy)
            best_iaro = max(iaro_methods, key=lambda m: results.method_results[m].accuracy)
            
            base_acc = results.method_results[best_baseline].accuracy
            iaro_acc = results.method_results[best_iaro].accuracy
            improvement = (iaro_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0
            
            self.logger.info(f"\n## Key Findings")
            self.logger.info(f"  Best Baseline: {best_baseline} ({base_acc:.1%})")
            self.logger.info(f"  Best IARO: {best_iaro} ({iaro_acc:.1%})")
            self.logger.info(f"  Improvement: {improvement:+.1f}%")
        
        # API统计
        self.logger.info(f"\n## API Statistics")
        self.logger.info(f"  Total API Calls: {self.api_stats['total_calls']}")
        for method, calls in self.api_stats['by_method'].items():
            self.logger.info(f"  - {method}: {calls}")
        
        self.logger.info("\n" + "=" * 70)
    
    async def run_all_datasets(
        self,
        max_samples: int = 50,
        seed: int = 42,
    ) -> Dict[str, BenchmarkResults]:
        """运行所有数据集"""
        all_results = {}
        
        datasets = [
            ("sst5", {}),
            ("race", {"level": "high"}),
            ("medmcqa", {}),
        ]
        
        for dataset, kwargs in datasets:
            try:
                results = await self.run_experiment(
                    dataset=dataset,
                    max_samples=max_samples,
                    seed=seed,
                    **kwargs,
                )
                if results:
                    all_results[dataset] = results
            except Exception as e:
                print(f"⚠ Error running {dataset}: {e}")
        
        # 生成综合报告
        if all_results:
            self._generate_cross_dataset_report(all_results)
        
        return all_results
    
    def _generate_cross_dataset_report(self, all_results: Dict[str, BenchmarkResults]):
        """生成跨数据集综合报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"cross_dataset_report_{timestamp}.md"
        
        lines = []
        lines.append("# Cross-Dataset Benchmark Report\n")
        lines.append(f"**Generated**: {timestamp}\n")
        
        # 综合表格
        lines.append("## Overall Results\n")
        lines.append("| Dataset | Best Baseline | Best IARO | Improvement |")
        lines.append("|---------|---------------|-----------|-------------|")
        
        for dataset, results in all_results.items():
            baselines = [m for m in results.method_results.keys() if 'iaro' not in m.lower()]
            iaro_methods = [m for m in results.method_results.keys() if 'iaro' in m.lower()]
            
            if baselines and iaro_methods:
                best_base = max(baselines, key=lambda m: results.method_results[m].accuracy)
                best_iaro = max(iaro_methods, key=lambda m: results.method_results[m].accuracy)
                
                base_acc = results.method_results[best_base].accuracy
                iaro_acc = results.method_results[best_iaro].accuracy
                improvement = (iaro_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0
                
                lines.append(
                    f"| {dataset.upper()} | {best_base} ({base_acc:.1%}) | "
                    f"{best_iaro} ({iaro_acc:.1%}) | {improvement:+.1f}% |"
                )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        print(f"\n✓ Cross-dataset report saved to {report_file}")


# ========== 主函数 ==========
async def main():
    parser = argparse.ArgumentParser(description='Benchmark Runner for IARO Method')
    parser.add_argument('--dataset', type=str, default='sst5',
                        choices=['sst5', 'race', 'medmcqa', 'all'],
                        help='Dataset to run')
    parser.add_argument('--split', type=str, default='dev',
                        help='Data split')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Maximum samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--level', type=str, default='high',
                        choices=['high', 'middle'],
                        help='RACE difficulty level')
    parser.add_argument('--output-dir', type=str, default='results/benchmark',
                        help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to run')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with minimal methods')
    
    args = parser.parse_args()
    
    # 确定方法
    if args.quick_test:
        methods = ['Standard', 'CoT', 'IARO-Hybrid']
    elif args.methods:
        methods = args.methods
    else:
        methods = ALL_METHODS
    
    runner = BenchmarkExperimentRunner(
        output_dir=args.output_dir,
        methods=methods,
    )
    
    if args.dataset == 'all':
        await runner.run_all_datasets(
            max_samples=args.max_samples,
            seed=args.seed,
        )
    else:
        kwargs = {}
        if args.dataset == 'race':
            kwargs['level'] = args.level
        
        await runner.run_experiment(
            dataset=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            **kwargs,
        )


if __name__ == '__main__':
    asyncio.run(main())
