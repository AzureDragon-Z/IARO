"""
评估器模块 (evaluator.py)
实现 ICR (Implicit Constraint Recall) 指标计算

用于ACL论文: Bridging the Intent Gap: Multi-Faceted Intent Recognition and Inference-Time Prompt Optimization

核心指标:
- ICR (Implicit Constraint Recall): 隐式需求召回率
- Goal Completion Rate: 显式目标完成率
- Style Adherence: 风格一致性
- Overall Quality: 综合质量评分

Cross-Model Evaluation:
- 被测模型是 DeepSeek -> 用 GPT-4o 做 Judge
- 被测模型是 GPT-4o -> 用 DeepSeek/Claude 做 Judge
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

from config import config
from llm_client import AsyncLLMClient, create_judge_client
from dataset_gen import DataSample
from methods import SolverResponse

# ========== 日志配置 ==========
def setup_eval_logger(log_file: str = None) -> logging.Logger:
    """设置评估日志"""
    logger = logging.getLogger("evaluator")
    logger.setLevel(logging.INFO)
    
    # 清除现有handlers
    logger.handlers.clear()
    
    # 文件handler
    if log_file is None:
        log_dir = Path("logs/evaluation")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    # 控制台handler (只显示重要信息)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger

# 全局logger
eval_logger = setup_eval_logger()


# ========== 评估结果数据结构 ==========
@dataclass
class NeedEvaluation:
    """单个需求的评估结果"""
    need_id: str
    need_description: str
    need_category: str
    need_importance: str
    is_addressed: bool
    confidence: float  # 0-1
    evidence: str  # 从响应中提取的证据


@dataclass
class SampleEvaluation:
    """单个样本的评估结果"""
    sample_id: str
    method: str
    query: str
    
    # ICR 核心指标
    implicit_needs_total: int
    implicit_needs_addressed: int
    icr_score: float  # Implicit Constraint Recall
    
    # 辅助指标
    explicit_goals_total: int
    explicit_goals_addressed: int
    goal_completion_rate: float
    
    style_adherence_score: float  # 0-1
    overall_quality_score: float  # 1-5
    
    # 详细评估
    need_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    
    # 元数据
    judge_model: str = ""
    evaluation_latency_ms: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResults:
    """实验结果汇总"""
    experiment_id: str
    dataset_name: str
    num_samples: int
    
    # 各方法的平均指标
    method_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 详细样本结果
    sample_evaluations: List[SampleEvaluation] = field(default_factory=list)
    
    # 统计信息
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    judge_model: str = ""
    solver_model: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["sample_evaluations"] = [e.to_dict() for e in self.sample_evaluations]
        return result


# ========== Judge Prompts ==========

# ICR 评估 Prompt
ICR_EVALUATION_PROMPT = """You are an expert evaluator assessing whether an AI response adequately addresses implicit user needs.

## Context:
A user submitted a request, and an AI assistant provided a response. The user had certain IMPLICIT NEEDS that were NOT explicitly mentioned in their query but are essential for a quality response.

## User Query:
{query}

## Ground Truth Implicit Needs (that should be addressed):
{implicit_needs}

## AI Response to Evaluate:
{response}

## Your Task:
For EACH implicit need listed above, determine if the AI response adequately addresses it.

## Evaluation Criteria:
- **Addressed (YES)**: The response clearly handles this need, either explicitly or implicitly through its implementation/content
- **Not Addressed (NO)**: The response ignores or inadequately handles this need

## Output Format (JSON):
{{
    "evaluations": [
        {{
            "need_id": "N1",
            "is_addressed": true/false,
            "confidence": 0.0-1.0,
            "evidence": "Quote or describe the part of response that addresses this (or explain why it's missing)"
        }},
        ...
    ],
    "summary": {{
        "total_needs": <int>,
        "addressed_count": <int>,
        "icr_score": <float 0-1>
    }}
}}

Be strict but fair. Only mark as "addressed" if the response genuinely handles the need."""


# 显式目标评估 Prompt
GOAL_EVALUATION_PROMPT = """You are an expert evaluator assessing goal completion.

## User Query:
{query}

## Explicit Goals:
{explicit_goals}

## AI Response:
{response}

## Task:
Evaluate if each explicit goal is achieved.

## Output Format (JSON):
{{
    "evaluations": [
        {{"goal_id": "G1", "is_achieved": true/false, "explanation": "..."}},
        ...
    ],
    "goal_completion_rate": <float 0-1>
}}"""


# 综合质量评估 Prompt
QUALITY_EVALUATION_PROMPT = """You are an expert evaluator assessing response quality.

## User Query:
{query}

## Style Constraints (if any):
{style_constraints}

## AI Response:
{response}

## Evaluation Dimensions:

1. **Style Adherence** (0-1): Does the response match expected tone, format, and presentation?
2. **Completeness** (1-5): How thoroughly does the response address the request?
3. **Clarity** (1-5): How clear and well-structured is the response?
4. **Usefulness** (1-5): How practically useful would this response be?

## Output Format (JSON):
{{
    "style_adherence": <float 0-1>,
    "completeness": <int 1-5>,
    "clarity": <int 1-5>,
    "usefulness": <int 1-5>,
    "overall_quality": <float 1-5>,
    "strengths": ["...", ...],
    "weaknesses": ["...", ...]
}}"""


# ========== 评估器类 ==========
class GPTEvaluator:
    """
    GPT-based Evaluator for ICR metrics
    
    使用 Cross-Model Evaluation 策略:
    - 被测模型是 DeepSeek -> 用 GPT-4o 做 Judge
    - 被测模型是 GPT-4o -> 用 DeepSeek 做 Judge
    """
    
    def __init__(
        self,
        judge_client: Optional[AsyncLLMClient] = None,
        output_dir: str = "results/evaluations",
    ):
        """
        初始化评估器
        
        Args:
            judge_client: 评判模型客户端
            output_dir: 输出目录
        """
        self.judge = judge_client or create_judge_client()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def evaluate_icr(
        self,
        query: str,
        response: str,
        implicit_needs: List[Dict[str, Any]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        评估隐式需求召回率 (ICR)
        
        Args:
            query: 用户查询
            response: 模型响应
            implicit_needs: Ground Truth 隐式需求列表
        
        Returns:
            (ICR分数, 详细评估列表)
        """
        # 格式化隐式需求
        needs_text = "\n".join(
            f"- [{n.get('id', f'N{i}')}] [{n.get('category', 'general')}] "
            f"[{n.get('importance', 'important')}]: {n.get('description', str(n))}"
            for i, n in enumerate(implicit_needs, 1)
        )
        
        prompt = ICR_EVALUATION_PROMPT.format(
            query=query,
            implicit_needs=needs_text,
            response=response[:3000],  # 截断以避免过长
        )
        
        result = await self.judge.generate_json(
            prompt=prompt,
            temperature=config.evaluation.judge_temperature,
            max_tokens=1500,
        )
        
        # 提取结果 (处理不同模型的响应格式)
        evaluations = result.get("evaluations", [])
        summary = result.get("summary", {})
        
        # 确保 evaluations 是字典列表
        parsed_evaluations = []
        for e in evaluations:
            if isinstance(e, dict):
                parsed_evaluations.append(e)
            elif isinstance(e, str):
                # 如果是字符串，尝试解析或创建默认结构
                parsed_evaluations.append({
                    "need_id": "unknown",
                    "is_addressed": "yes" in e.lower() or "true" in e.lower(),
                    "confidence": 0.5,
                    "evidence": e
                })
        
        evaluations = parsed_evaluations
        
        # 计算 ICR 分数
        icr_score = summary.get("icr_score", 0.0) if isinstance(summary, dict) else 0.0
        if (icr_score == 0.0 or not isinstance(icr_score, (int, float))) and evaluations:
            # 手动计算
            addressed = sum(1 for e in evaluations if e.get("is_addressed", False))
            icr_score = addressed / len(evaluations) if evaluations else 0.0
        
        return icr_score, evaluations
    
    async def evaluate_goals(
        self,
        query: str,
        response: str,
        explicit_goals: List[Any],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        评估显式目标完成率
        
        Args:
            query: 用户查询
            response: 模型响应
            explicit_goals: 显式目标列表
        
        Returns:
            (完成率, 详细评估列表)
        """
        # 格式化目标
        goals_text = "\n".join(
            f"- [G{i}]: {g.get('description', str(g)) if isinstance(g, dict) else g}"
            for i, g in enumerate(explicit_goals, 1)
        )
        
        prompt = GOAL_EVALUATION_PROMPT.format(
            query=query,
            explicit_goals=goals_text,
            response=response[:3000],
        )
        
        result = await self.judge.generate_json(
            prompt=prompt,
            temperature=config.evaluation.judge_temperature,
            max_tokens=1000,
        )
        
        evaluations = result.get("evaluations", [])
        completion_rate = result.get("goal_completion_rate", 0.0)
        
        # 确保 evaluations 是字典列表
        parsed_evaluations = []
        for e in evaluations:
            if isinstance(e, dict):
                parsed_evaluations.append(e)
            elif isinstance(e, str):
                parsed_evaluations.append({
                    "goal_id": "unknown",
                    "is_achieved": "yes" in e.lower() or "true" in e.lower(),
                    "explanation": e
                })
        evaluations = parsed_evaluations
        
        if (completion_rate == 0.0 or not isinstance(completion_rate, (int, float))) and evaluations:
            achieved = sum(1 for e in evaluations if e.get("is_achieved", False))
            completion_rate = achieved / len(evaluations) if evaluations else 0.0
        
        return completion_rate, evaluations
    
    async def evaluate_quality(
        self,
        query: str,
        response: str,
        style_constraints: List[Any] = None,
    ) -> Dict[str, Any]:
        """
        评估综合质量
        
        Args:
            query: 用户查询
            response: 模型响应
            style_constraints: 风格约束
        
        Returns:
            质量评估字典
        """
        style_text = "None specified"
        if style_constraints:
            style_text = "\n".join(
                f"- {s.get('description', str(s)) if isinstance(s, dict) else s}"
                for s in style_constraints
            )
        
        prompt = QUALITY_EVALUATION_PROMPT.format(
            query=query,
            style_constraints=style_text,
            response=response[:3000],
        )
        
        result = await self.judge.generate_json(
            prompt=prompt,
            temperature=config.evaluation.judge_temperature,
            max_tokens=800,
        )
        
        # 确保返回有效的质量评估字典
        if not isinstance(result, dict):
            result = {}
        
        # 设置默认值
        defaults = {
            "style_adherence": 0.5,
            "completeness": 3,
            "clarity": 3,
            "usefulness": 3,
            "overall_quality": 3.0,
            "strengths": [],
            "weaknesses": []
        }
        
        for key, default_val in defaults.items():
            if key not in result or not isinstance(result.get(key), (int, float, list)):
                result[key] = default_val
        
        return result
    
    async def evaluate_sample(
        self,
        sample: DataSample,
        response: SolverResponse,
    ) -> SampleEvaluation:
        """
        评估单个样本
        
        Args:
            sample: 数据样本 (含 Ground Truth)
            response: 求解器响应
        
        Returns:
            SampleEvaluation
        """
        import time
        start = time.time()
        
        eval_logger.info(f"[START] {response.method} | {sample.id} | Query: {sample.ambiguous_query[:50]}...")
        
        try:
            # 1. 评估 ICR
            eval_logger.info(f"  [ICR] Evaluating implicit needs for {sample.id}...")
            icr_score, icr_details = await self.evaluate_icr(
                query=sample.ambiguous_query,
                response=response.response,
                implicit_needs=sample.implicit_needs,
            )
            eval_logger.info(f"  [ICR] {sample.id} -> Score: {icr_score:.2f}")
        except Exception as e:
            eval_logger.error(f"  [ICR] FAILED for {sample.id}: {e}")
            icr_score, icr_details = 0.0, []
        
        try:
            # 2. 评估目标完成率
            eval_logger.info(f"  [GOAL] Evaluating explicit goals for {sample.id}...")
            goal_rate, goal_details = await self.evaluate_goals(
                query=sample.ambiguous_query,
                response=response.response,
                explicit_goals=sample.explicit_goals,
            )
            eval_logger.info(f"  [GOAL] {sample.id} -> Rate: {goal_rate:.2f}")
        except Exception as e:
            eval_logger.error(f"  [GOAL] FAILED for {sample.id}: {e}")
            goal_rate, goal_details = 0.0, []
        
        try:
            # 3. 评估综合质量
            eval_logger.info(f"  [QUALITY] Evaluating overall quality for {sample.id}...")
            quality = await self.evaluate_quality(
                query=sample.ambiguous_query,
                response=response.response,
                style_constraints=sample.style_constraints,
            )
            eval_logger.info(f"  [QUALITY] {sample.id} -> Score: {quality.get('overall_quality', 0)}")
        except Exception as e:
            eval_logger.error(f"  [QUALITY] FAILED for {sample.id}: {e}")
            quality = {"style_adherence": 0.5, "overall_quality": 3.0}
        
        latency = (time.time() - start) * 1000
        eval_logger.info(f"[DONE] {response.method} | {sample.id} | ICR={icr_score:.2f} | Goal={goal_rate:.2f} | {latency:.0f}ms")
        
        return SampleEvaluation(
            sample_id=sample.id,
            method=response.method,
            query=sample.ambiguous_query,
            implicit_needs_total=len(sample.implicit_needs),
            implicit_needs_addressed=int(icr_score * len(sample.implicit_needs)),
            icr_score=icr_score,
            explicit_goals_total=len(sample.explicit_goals),
            explicit_goals_addressed=int(goal_rate * len(sample.explicit_goals)),
            goal_completion_rate=goal_rate,
            style_adherence_score=quality.get("style_adherence", 0.0),
            overall_quality_score=quality.get("overall_quality", 0.0),
            need_evaluations=icr_details,
            judge_model=self.judge.model,
            evaluation_latency_ms=latency,
            timestamp=datetime.now().isoformat(),
        )
    
    async def evaluate_experiment(
        self,
        samples: List[DataSample],
        responses: Dict[str, List[SolverResponse]],
        experiment_id: Optional[str] = None,
    ) -> ExperimentResults:
        """
        评估完整实验
        
        Args:
            samples: 数据样本列表
            responses: 方法名到响应列表的映射
            experiment_id: 实验ID
        
        Returns:
            ExperimentResults
        """
        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"Evaluating Experiment: {experiment_id}")
        print(f"{'='*60}")
        print(f"Samples: {len(samples)}")
        print(f"Methods: {list(responses.keys())}")
        print(f"Judge: {self.judge.model}")
        print(f"{'='*60}\n")
        
        eval_logger.info(f"{'='*60}")
        eval_logger.info(f"EXPERIMENT START: {experiment_id}")
        eval_logger.info(f"Samples: {len(samples)} | Methods: {list(responses.keys())} | Judge: {self.judge.model}")
        eval_logger.info(f"{'='*60}")
        
        all_evaluations = []
        method_scores = {method: [] for method in responses.keys()}
        
        # 对每个方法的每个响应进行评估
        for method, method_responses in responses.items():
            print(f"\nEvaluating method: {method}")
            eval_logger.info(f"\n>>> METHOD: {method} | {len(method_responses)} samples")
            
            tasks = []
            for i, (sample, response) in enumerate(zip(samples, method_responses)):
                tasks.append(self.evaluate_sample(sample, response))
            
            # 并发评估
            results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"Evaluating {method}"
            )
            
            for eval_result in results:
                all_evaluations.append(eval_result)
                method_scores[method].append(eval_result)
            
            # 方法评估完成，记录汇总
            method_icr = sum(e.icr_score for e in results) / len(results) if results else 0
            eval_logger.info(f">>> METHOD COMPLETE: {method} | Avg ICR: {method_icr:.3f}")
        
        # 计算方法级别的统计
        method_results = {}
        for method, evals in method_scores.items():
            if not evals:
                continue
            
            icr_scores = [e.icr_score for e in evals]
            goal_rates = [e.goal_completion_rate for e in evals]
            quality_scores = [e.overall_quality_score for e in evals]
            
            method_results[method] = {
                "icr_mean": sum(icr_scores) / len(icr_scores),
                "icr_std": self._std(icr_scores),
                "goal_completion_mean": sum(goal_rates) / len(goal_rates),
                "goal_completion_std": self._std(goal_rates),
                "quality_mean": sum(quality_scores) / len(quality_scores),
                "quality_std": self._std(quality_scores),
                "num_samples": len(evals),
            }
        
        # 计算 IARO 相对于 Vanilla 的提升
        statistics = {}
        if "iaro" in method_results and "vanilla" in method_results:
            iaro_icr = method_results["iaro"]["icr_mean"]
            vanilla_icr = method_results["vanilla"]["icr_mean"]
            improvement = (iaro_icr - vanilla_icr) / vanilla_icr * 100 if vanilla_icr > 0 else 0
            statistics["iaro_vs_vanilla_icr_improvement"] = f"{improvement:.1f}%"
        
        results = ExperimentResults(
            experiment_id=experiment_id,
            dataset_name=config.experiment.dataset_name,
            num_samples=len(samples),
            method_results=method_results,
            sample_evaluations=all_evaluations,
            statistics=statistics,
            judge_model=self.judge.model,
            solver_model=responses.get("vanilla", [SolverResponse("", "", "")])[0].model if responses else "",
            timestamp=datetime.now().isoformat(),
        )
        
        # 打印摘要
        self._print_summary(results)
        
        return results
    
    def _std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _print_summary(self, results: ExperimentResults):
        """打印结果摘要"""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Experiment: {results.experiment_id}")
        print(f"Samples: {results.num_samples}")
        print(f"Judge Model: {results.judge_model}")
        print(f"{'='*60}")
        
        # 方法比较表
        print(f"\n{'Method':<15} {'ICR':>8} {'Goals':>8} {'Quality':>8}")
        print("-" * 45)
        
        for method, scores in sorted(results.method_results.items()):
            icr = f"{scores['icr_mean']:.3f}"
            goal = f"{scores['goal_completion_mean']:.3f}"
            qual = f"{scores['quality_mean']:.2f}"
            print(f"{method:<15} {icr:>8} {goal:>8} {qual:>8}")
        
        print("-" * 45)
        
        # 关键统计
        if results.statistics:
            print("\nKey Statistics:")
            for key, value in results.statistics.items():
                print(f"  {key}: {value}")
        
        print(f"\n{'='*60}")
    
    def save_results(
        self,
        results: ExperimentResults,
        filename: Optional[str] = None,
    ) -> Path:
        """保存评估结果"""
        if filename is None:
            filename = f"{results.experiment_id}_results.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"✓ Results saved to: {output_path}")
        return output_path


# ========== 统计检验工具 ==========
class StatisticalTests:
    """统计检验工具"""
    
    @staticmethod
    def paired_t_test(
        scores_a: List[float],
        scores_b: List[float],
    ) -> Dict[str, float]:
        """配对t检验"""
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
            return {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        except ImportError:
            # 手动计算
            n = len(scores_a)
            if n != len(scores_b) or n < 2:
                return {"error": "Invalid input"}
            
            diffs = [a - b for a, b in zip(scores_a, scores_b)]
            mean_diff = sum(diffs) / n
            std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)) ** 0.5
            
            if std_diff == 0:
                return {"error": "Zero variance"}
            
            t_stat = mean_diff / (std_diff / (n ** 0.5))
            
            return {
                "t_statistic": t_stat,
                "p_value": None,  # 需要scipy计算精确p值
                "mean_difference": mean_diff,
            }
    
    @staticmethod
    def cohens_d(
        scores_a: List[float],
        scores_b: List[float],
    ) -> float:
        """计算 Cohen's d 效应量"""
        n_a, n_b = len(scores_a), len(scores_b)
        mean_a = sum(scores_a) / n_a
        mean_b = sum(scores_b) / n_b
        
        var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1)
        
        pooled_std = ((var_a * (n_a - 1) + var_b * (n_b - 1)) / (n_a + n_b - 2)) ** 0.5
        
        if pooled_std == 0:
            return 0.0
        
        return (mean_a - mean_b) / pooled_std
    
    @staticmethod
    def win_rate(
        scores_a: List[float],
        scores_b: List[float],
    ) -> Dict[str, float]:
        """计算胜率"""
        wins_a = sum(1 for a, b in zip(scores_a, scores_b) if a > b)
        wins_b = sum(1 for a, b in zip(scores_a, scores_b) if b > a)
        ties = sum(1 for a, b in zip(scores_a, scores_b) if a == b)
        total = len(scores_a)
        
        return {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": wins_a / total if total > 0 else 0,
            "win_rate_b": wins_b / total if total > 0 else 0,
        }


# ========== 测试代码 ==========
async def test_evaluator():
    """测试评估器"""
    print("=" * 60)
    print("Testing GPTEvaluator")
    print("=" * 60)
    
    evaluator = GPTEvaluator()
    
    # 模拟测试数据
    test_query = "Write a Python script to process CSV files"
    
    test_response_good = """
Here's a Python script to process CSV files:

```python
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_csv(file_path: str) -> List[Dict]:
    \"\"\"Process a CSV file and return its contents as a list of dictionaries.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries representing rows
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or invalid
    \"\"\"
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not path.suffix.lower() == '.csv':
        logger.warning(f"File may not be a CSV: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            
        if not data:
            raise ValueError("CSV file is empty")
            
        logger.info(f"Processed {len(data)} rows from {file_path}")
        return data
        
    except csv.Error as e:
        logger.error(f"CSV parsing error: {e}")
        raise
```

This script includes error handling, logging, and type hints.
    """
    
    test_response_bad = """
Here's how to process CSV:

```python
import csv
f = open('file.csv')
data = csv.reader(f)
for row in data:
    print(row)
```
    """
    
    implicit_needs = [
        {"id": "N1", "description": "Error handling for file operations", "category": "safety", "importance": "critical"},
        {"id": "N2", "description": "Input validation", "category": "functional", "importance": "important"},
        {"id": "N3", "description": "Proper logging", "category": "quality", "importance": "important"},
        {"id": "N4", "description": "Type hints and documentation", "category": "quality", "importance": "nice_to_have"},
    ]
    
    # 测试 ICR 评估
    print("\n1. Testing ICR evaluation (good response)...")
    icr_good, details_good = await evaluator.evaluate_icr(
        query=test_query,
        response=test_response_good,
        implicit_needs=implicit_needs,
    )
    print(f"ICR Score (good): {icr_good:.3f}")
    
    print("\n2. Testing ICR evaluation (bad response)...")
    icr_bad, details_bad = await evaluator.evaluate_icr(
        query=test_query,
        response=test_response_bad,
        implicit_needs=implicit_needs,
    )
    print(f"ICR Score (bad): {icr_bad:.3f}")
    
    print(f"\n✓ Difference: {icr_good - icr_bad:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_evaluator())
