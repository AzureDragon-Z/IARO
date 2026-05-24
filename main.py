"""
主实验脚本 (main.py)
编排整个实验流程：生成数据 -> 运行 Solver -> 评估 -> 输出报告

用于ACL论文: Bridging the Intent Gap: Multi-Faceted Intent Recognition and Inference-Time Prompt Optimization

Usage:
    python main.py --mode full          # 完整实验
    python main.py --mode generate      # 仅生成数据
    python main.py --mode evaluate      # 仅评估 (需要已有响应)
    python main.py --mode quick_test    # 快速测试 (小样本)
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from config import config
from llm_client import AsyncLLMClient, create_solver_client, create_judge_client, create_generator_client
from dataset_gen import AmbiguBenchGenerator, DataSample, load_dataset, save_dataset
from methods import SolverFactory, BaseSolver, SolverResponse
from evaluator import GPTEvaluator, ExperimentResults, StatisticalTests


# ========== 实验配置 ==========
class ExperimentRunner:
    """实验运行器"""
    
    def __init__(
        self,
        experiment_name: str = None,
        output_dir: str = "results/api_experiments",
        methods: List[str] = None,
        solver_provider: str = None,  # 指定solver使用的provider (local, siliconflow等)
        solver_model: str = None,     # 指定solver使用的模型
        solver_base_url: str = None,  # 本地模型的base_url
    ):
        """
        初始化实验运行器
        
        Args:
            experiment_name: 实验名称
            output_dir: 输出目录
            methods: 要运行的方法列表
            solver_provider: Solver使用的provider (默认使用config中的配置)
            solver_model: Solver使用的模型 (默认使用config中的配置)
            solver_base_url: 本地模型的base_url (仅当solver_provider='local'时使用)
        """
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.methods = methods or ["vanilla", "cot", "self_refine", "iaro"]
        
        # Solver配置
        self.solver_provider = solver_provider
        self.solver_model = solver_model
        self.solver_base_url = solver_base_url
        
        # 客户端
        self.solver_client = None
        self.judge_client = None
        self.generator_client = None
        
        # 数据
        self.samples: List[DataSample] = []
        self.responses: Dict[str, List[SolverResponse]] = {}
        self.results: Optional[ExperimentResults] = None
        
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Output: {self.output_dir}")
        print(f"Methods: {self.methods}")
        print(f"{'='*60}\n")
    
    def _init_clients(self):
        """初始化客户端"""
        if self.solver_client is None:
            if self.solver_provider:
                # 使用指定的provider创建solver客户端
                self.solver_client = AsyncLLMClient(
                    provider=self.solver_provider,
                    model=self.solver_model,
                    base_url=self.solver_base_url,
                )
                print(f"✓ Using custom solver: {self.solver_provider} / {self.solver_model}")
            else:
                self.solver_client = create_solver_client()
        if self.judge_client is None:
            # Judge始终使用DeepSeek V3
            self.judge_client = create_judge_client()
        if self.generator_client is None:
            self.generator_client = create_generator_client()
    
    async def generate_dataset(
        self,
        num_samples: int = 50,
        output_file: str = "dataset.jsonl",
    ) -> List[DataSample]:
        """
        生成 AmbiguBench 数据集
        
        Args:
            num_samples: 样本数量
            output_file: 输出文件名
        
        Returns:
            生成的样本列表
        """
        self._init_clients()
        
        generator = AmbiguBenchGenerator(
            generator_client=self.generator_client,
            output_dir=str(self.output_dir / "data"),
        )
        
        samples = await generator.generate_dataset(
            num_samples=num_samples,
            output_file=output_file,
        )
        
        self.samples = samples
        return samples
    
    def load_dataset(self, file_path: str) -> List[DataSample]:
        """加载已有数据集"""
        self.samples = load_dataset(file_path)
        print(f"✓ Loaded {len(self.samples)} samples from {file_path}")
        return self.samples
    
    async def run_solvers(
        self,
        samples: List[DataSample] = None,
        methods: List[str] = None,
    ) -> Dict[str, List[SolverResponse]]:
        """
        运行所有求解器
        
        Args:
            samples: 数据样本
            methods: 方法列表
        
        Returns:
            方法名到响应列表的映射
        """
        self._init_clients()
        
        samples = samples or self.samples
        methods = methods or self.methods
        
        if not samples:
            raise ValueError("No samples to solve. Generate or load dataset first.")
        
        print(f"\n{'='*60}")
        print(f"Running Solvers")
        print(f"{'='*60}")
        print(f"Samples: {len(samples)}")
        print(f"Methods: {methods}")
        print(f"Solver Model: {self.solver_client.model}")
        print(f"{'='*60}\n")
        
        # 提取查询
        queries = [s.ambiguous_query for s in samples]
        
        # 创建求解器
        solvers = SolverFactory.create_all(
            client=self.solver_client,
            methods=methods,
        )
        
        # 运行每个方法
        responses = {}
        for method_name, solver in solvers.items():
            print(f"\n>>> Running {method_name}...")
            method_responses = await solver.solve_batch(queries, show_progress=True)
            responses[method_name] = method_responses
            
            # 保存中间结果
            self._save_responses(method_name, method_responses)
        
        self.responses = responses
        return responses
    
    def _save_responses(self, method: str, responses: List[SolverResponse]):
        """保存方法响应"""
        output_file = self.output_dir / "responses" / f"{method}_responses.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in responses:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"  ✓ Saved to {output_file}")
    
    def load_responses(self, responses_dir: str) -> Dict[str, List[SolverResponse]]:
        """
        从目录加载已保存的响应
        
        Args:
            responses_dir: 响应文件目录
        
        Returns:
            方法名到响应列表的映射
        """
        responses_path = Path(responses_dir)
        responses = {}
        
        for response_file in responses_path.glob("*_responses.jsonl"):
            method = response_file.stem.replace("_responses", "")
            method_responses = []
            
            with open(response_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        method_responses.append(SolverResponse(**data))
            
            responses[method] = method_responses
            print(f"✓ Loaded {len(method_responses)} responses for {method}")
        
        self.responses = responses
        return responses
    
    async def evaluate(
        self,
        samples: List[DataSample] = None,
        responses: Dict[str, List[SolverResponse]] = None,
    ) -> ExperimentResults:
        """
        评估实验结果
        
        Args:
            samples: 数据样本
            responses: 方法响应
        
        Returns:
            实验结果
        """
        self._init_clients()
        
        samples = samples or self.samples
        responses = responses or self.responses
        
        if not samples or not responses:
            raise ValueError("No samples or responses to evaluate.")
        
        evaluator = GPTEvaluator(
            judge_client=self.judge_client,
            output_dir=str(self.output_dir / "evaluations"),
        )
        
        results = await evaluator.evaluate_experiment(
            samples=samples,
            responses=responses,
            experiment_id=self.experiment_name,
        )
        
        # 保存结果
        evaluator.save_results(results)
        
        self.results = results
        return results
    
    def generate_report(
        self,
        results: ExperimentResults = None,
    ) -> pd.DataFrame:
        """
        生成实验报告
        
        Args:
            results: 实验结果
        
        Returns:
            结果 DataFrame
        """
        results = results or self.results
        
        if not results:
            raise ValueError("No results to report.")
        
        # 创建方法比较表
        rows = []
        for method, scores in results.method_results.items():
            rows.append({
                "Method": method.upper(),
                "ICR": f"{scores['icr_mean']:.3f} ± {scores['icr_std']:.3f}",
                "ICR_mean": scores['icr_mean'],
                "Goal Completion": f"{scores['goal_completion_mean']:.3f} ± {scores['goal_completion_std']:.3f}",
                "Quality": f"{scores['quality_mean']:.2f} ± {scores['quality_std']:.2f}",
                "N": scores['num_samples'],
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("ICR_mean", ascending=False)
        
        # 保存CSV
        csv_path = self.output_dir / "results_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        
        # 生成 LaTeX 表格
        latex_table = self._generate_latex_table(results)
        latex_path = self.output_dir / "results_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"✓ LaTeX table saved to {latex_path}")
        
        # 统计检验
        self._run_statistical_tests(results)
        
        return df
    
    def _generate_latex_table(self, results: ExperimentResults) -> str:
        """生成 LaTeX 表格"""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Comparison of methods on AmbiguBench-SOTA}",
            r"\label{tab:main_results}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Method & ICR$\uparrow$ & Goal Completion$\uparrow$ & Quality$\uparrow$ \\",
            r"\midrule",
        ]
        
        # 按 ICR 排序
        sorted_methods = sorted(
            results.method_results.items(),
            key=lambda x: x[1]['icr_mean'],
            reverse=True
        )
        
        best_icr = sorted_methods[0][1]['icr_mean'] if sorted_methods else 0
        
        for method, scores in sorted_methods:
            icr = scores['icr_mean']
            icr_str = f"{icr:.3f}"
            
            # 最佳结果加粗
            if icr == best_icr:
                icr_str = r"\textbf{" + icr_str + "}"
            
            goal = f"{scores['goal_completion_mean']:.3f}"
            quality = f"{scores['quality_mean']:.2f}"
            
            method_display = method.upper()
            if method == "iaro":
                method_display = r"\textbf{IARO (Ours)}"
            
            lines.append(f"{method_display} & {icr_str} & {goal} & {quality} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _run_statistical_tests(self, results: ExperimentResults):
        """运行统计检验"""
        print("\n" + "="*60)
        print("Statistical Tests")
        print("="*60)
        
        # 提取 IARO 和 Vanilla 的 ICR 分数
        iaro_scores = []
        vanilla_scores = []
        
        for eval_result in results.sample_evaluations:
            if eval_result.method == "iaro":
                iaro_scores.append(eval_result.icr_score)
            elif eval_result.method == "vanilla":
                vanilla_scores.append(eval_result.icr_score)
        
        if len(iaro_scores) == len(vanilla_scores) and len(iaro_scores) > 1:
            # 配对 t 检验
            t_test = StatisticalTests.paired_t_test(iaro_scores, vanilla_scores)
            print(f"\nPaired t-test (IARO vs Vanilla):")
            print(f"  t-statistic: {t_test.get('t_statistic', 'N/A'):.4f}")
            print(f"  p-value: {t_test.get('p_value', 'N/A')}")
            
            # Cohen's d
            d = StatisticalTests.cohens_d(iaro_scores, vanilla_scores)
            print(f"  Cohen's d: {d:.4f}")
            
            # 胜率
            win_rate = StatisticalTests.win_rate(iaro_scores, vanilla_scores)
            print(f"\nWin Rate:")
            print(f"  IARO wins: {win_rate['wins_a']} ({win_rate['win_rate_a']:.1%})")
            print(f"  Vanilla wins: {win_rate['wins_b']} ({win_rate['win_rate_b']:.1%})")
            print(f"  Ties: {win_rate['ties']}")
        
        print("="*60)
    
    async def _run_experiment_after_data_loaded(self):
        """
        数据已加载后运行实验 (用于随机采样场景)
        """
        print("\n" + "="*70)
        print("EXPERIMENT PIPELINE (Data Pre-loaded)")
        print("="*70)
        print(f"Experiment: {self.experiment_name}")
        print(f"Samples: {len(self.samples)}")
        print(f"Methods: {self.methods}")
        print("="*70 + "\n")
        
        # Step 1: 运行求解器
        print("\n" + "="*60)
        print("STEP 1: Run Solvers")
        print("="*60)
        
        await self.run_solvers()
        
        # Step 2: 评估
        print("\n" + "="*60)
        print("STEP 2: Evaluate Results")
        print("="*60)
        
        await self.evaluate()
        
        # Step 3: 生成报告
        print("\n" + "="*60)
        print("STEP 3: Generate Report")
        print("="*60)
        
        df = self.generate_report()
        
        # 打印最终结果
        self._print_final_results(df)
    
    async def run_full_experiment(
        self,
        num_samples: int = 50,
        generate_data: bool = True,
        data_file: str = None,
    ):
        """
        运行完整实验流程
        
        Args:
            num_samples: 样本数量
            generate_data: 是否生成新数据
            data_file: 已有数据文件路径
        """
        # Step 1: 准备数据
        print("\n" + "="*60)
        print("STEP 1: Prepare Dataset")
        print("="*60)
        
        if generate_data:
            await self.generate_dataset(num_samples=num_samples)
        elif data_file:
            self.load_dataset(data_file)
        else:
            raise ValueError("Either generate_data or data_file must be specified")
        
        print("\n" + "="*70)
        print("FULL EXPERIMENT PIPELINE")
        print("="*70)
        print(f"Experiment: {self.experiment_name}")
        print(f"Samples: {len(self.samples)}")
        print(f"Methods: {self.methods}")
        print("="*70 + "\n")
        
        # Step 2: 运行求解器
        print("\n" + "="*60)
        print("STEP 2: Run Solvers")
        print("="*60)
        
        await self.run_solvers()
        
        # Step 3: 评估
        print("\n" + "="*60)
        print("STEP 3: Evaluate Results")
        print("="*60)
        
        await self.evaluate()
        
        # Step 4: 生成报告
        print("\n" + "="*60)
        print("STEP 4: Generate Report")
        print("="*60)
        
        df = self.generate_report()
        
        # 打印最终结果
        self._print_final_results(df)
        
        # 保存完整实验配置
        self._save_experiment_config()
        
        return df
    
    def _print_final_results(self, df: pd.DataFrame):
        """打印最终结果"""
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
    
    def _save_experiment_config(self):
        """保存实验配置"""
        config_data = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "methods": self.methods,
            "num_samples": len(self.samples),
            "solver_model": self.solver_client.model if self.solver_client else None,
            "judge_model": self.judge_client.model if self.judge_client else None,
            "config": config.to_dict(),
        }
        
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Experiment config saved to {config_path}")


# ========== 快速测试函数 ==========
async def quick_test(
    num_samples: int = 5,
    methods: List[str] = None,
    generate_data: bool = False,
    data_file: str = None,
    solver_provider: str = None,
    solver_model: str = None,
    solver_base_url: str = None,
    random_sample: bool = False,
):
    """
    快速测试
    
    Args:
        num_samples: 样本数量
        methods: 要运行的方法
        generate_data: 是否强制生成新数据
        data_file: 已有数据文件路径
        solver_provider: Solver使用的provider
        solver_model: Solver使用的模型
        solver_base_url: 本地模型的base_url
        random_sample: 是否从现有数据集随机采样
    """
    import random as rand_module
    
    methods = methods or ["vanilla", "iaro"]
    
    runner = ExperimentRunner(
        experiment_name=f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        methods=methods,
        solver_provider=solver_provider,
        solver_model=solver_model,
        solver_base_url=solver_base_url,
    )
    
    # 智能判断是否需要生成数据
    should_generate = generate_data
    actual_data_file = data_file
    
    if not generate_data and not data_file:
        # 尝试查找最近的数据集 (不限制最小样本数，因为我们会随机采样)
        recent_dataset = find_recent_dataset(min_samples=0)
        if recent_dataset:
            print(f"✓ Found existing dataset: {recent_dataset}")
            print(f"  Use --generate-data to force regeneration")
            actual_data_file = recent_dataset
        else:
            print("⚠ No existing dataset found, generating new one...")
            should_generate = True
    
    # 如果使用随机采样，先加载数据集再采样
    if random_sample and actual_data_file and not should_generate:
        runner.load_dataset(actual_data_file)
        total_samples = len(runner.samples)
        
        if num_samples < total_samples:
            print(f"\n🎲 Random sampling {num_samples} from {total_samples} samples...")
            rand_module.seed(42)  # 固定种子以便复现
            runner.samples = rand_module.sample(runner.samples, num_samples)
            print(f"✓ Sampled {len(runner.samples)} samples")
        
        # 直接运行，不需要再加载数据
        await runner._run_experiment_after_data_loaded()
    else:
        await runner.run_full_experiment(
            num_samples=num_samples,
            generate_data=should_generate,
            data_file=actual_data_file,
        )


async def generate_only(num_samples: int = 100):
    """仅生成数据集"""
    runner = ExperimentRunner(
        experiment_name=f"data_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    await runner.generate_dataset(num_samples=num_samples)
    print(f"\n✓ Dataset generated with {num_samples} samples")


async def full_experiment(
    num_samples: int = 100,
    methods: List[str] = None,
    generate_data: bool = False,
    data_file: str = None,
    solver_provider: str = None,
    solver_model: str = None,
    solver_base_url: str = None,
):
    """
    完整实验
    
    Args:
        num_samples: 样本数量
        methods: 要运行的方法
        generate_data: 是否强制生成新数据
        data_file: 已有数据文件路径
        solver_provider: Solver使用的provider
        solver_model: Solver使用的模型
        solver_base_url: 本地模型的base_url
    """
    methods = methods or ["vanilla", "cot", "self_refine", "iaro"]
    
    runner = ExperimentRunner(
        experiment_name=f"full_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        methods=methods,
        solver_provider=solver_provider,
        solver_model=solver_model,
        solver_base_url=solver_base_url,
    )
    
    # 智能判断是否需要生成数据
    should_generate = generate_data
    actual_data_file = data_file
    
    if not generate_data and not data_file:
        recent_dataset = find_recent_dataset(min_samples=num_samples)
        if recent_dataset:
            print(f"✓ Found existing dataset: {recent_dataset}")
            print(f"  Use --generate-data to force regeneration")
            actual_data_file = recent_dataset
        else:
            print("⚠ No existing dataset found, generating new one...")
            should_generate = True
    
    await runner.run_full_experiment(
        num_samples=num_samples,
        generate_data=should_generate,
        data_file=actual_data_file,
    )


async def run_on_dataset(data_file: str, methods: List[str] = None):
    """使用已有数据集运行实验"""
    methods = methods or ["vanilla", "cot", "self_refine", "iaro"]
    
    runner = ExperimentRunner(
        experiment_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        methods=methods,
    )
    
    await runner.run_full_experiment(
        num_samples=0,  # 不使用
        generate_data=False,
        data_file=data_file,
    )


async def evaluate_only(data_file: str, responses_dir: str, judge_model: str = None):
    """
    仅运行评估（从已保存的响应）
    
    Args:
        data_file: 数据集文件路径
        responses_dir: 响应文件目录
        judge_model: 指定Judge模型 (v3, r1, gpt4o)
    """
    # 从 responses_dir 推断实验目录
    responses_path = Path(responses_dir)
    experiment_dir = responses_path.parent
    
    # 根据judge模型调整实验名称
    if judge_model:
        experiment_name = f"{experiment_dir.name}_judge_{judge_model}"
    else:
        experiment_name = experiment_dir.name
    
    runner = ExperimentRunner(
        experiment_name=experiment_name,
        output_dir=str(experiment_dir.parent),
    )
    
    # 加载数据集
    runner.load_dataset(data_file)
    
    # 加载响应
    runner.load_responses(responses_dir)
    
    # 根据指定的judge模型创建客户端
    if judge_model:
        judge_configs = {
            "v3": ("siliconflow", "deepseek-ai/DeepSeek-V3"),
            "r1": ("siliconflow", "deepseek-ai/DeepSeek-R1"),
            "gpt4o": ("openai", "gpt-4o"),
        }
        if judge_model in judge_configs:
            provider, model = judge_configs[judge_model]
            runner.judge_client = AsyncLLMClient(
                provider=provider,
                model=model,
                semaphore_limit=3 if judge_model == "r1" else 5,  # R1较慢，降低并发
                timeout=300 if judge_model == "r1" else 120,      # R1需要更长超时
            )
            print(f"✓ 使用指定Judge模型: {model}")
    
    # 初始化其他客户端
    runner._init_clients()
    
    print(f"\n{'='*70}")
    print(f"EVALUATION ONLY MODE")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Samples: {len(runner.samples)}")
    print(f"Methods: {list(runner.responses.keys())}")
    print(f"Judge: {runner.judge_client.model}")
    print(f"{'='*70}\n")
    
    # 运行评估
    await runner.evaluate()
    
    # 生成报告
    runner.generate_report()


def find_recent_dataset(base_dir: str = "results/api_experiments", min_samples: int = 0) -> Optional[str]:
    """
    查找最近的满足样本数量要求的数据集文件
    
    Args:
        base_dir: 实验结果基础目录
        min_samples: 最小样本数量要求
    
    Returns:
        最近数据集的路径，如果没有找到则返回 None
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # 查找所有 dataset.jsonl 文件
    dataset_files = list(base_path.glob("*/data/dataset.jsonl"))
    
    if not dataset_files:
        return None
    
    # 按修改时间排序，返回最新的
    dataset_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # 如果没有样本数量要求，直接返回最新的
    if min_samples <= 0:
        return str(dataset_files[0])
    
    # 遍历文件查找满足数量要求的
    print(f"Searching for existing dataset with at least {min_samples} samples...")
    for file_path in dataset_files:
        try:
            # 快速计算行数
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    count += 1
            
            if count >= min_samples:
                return str(file_path)
            else:
                # Debug info (optional)
                # print(f"  Skipping {file_path.parent.parent.name}: {count} samples < {min_samples}")
                pass
        except Exception:
            continue
            
    return None


# ========== CLI 入口 ==========
def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="AmbiguBench Experiment Runner for ACL Paper"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="quick_test",
        choices=["full", "generate", "quick_test", "run", "evaluate"],
        help="Experiment mode: full, generate, run (use existing data), evaluate (from saved responses), quick_test"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to existing dataset file"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to run"
    )
    
    parser.add_argument(
        "--responses-dir",
        type=str,
        default=None,
        help="Path to directory containing saved responses (for evaluate mode)"
    )
    
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        choices=["v3", "r1", "gpt4o"],
        help="Judge model: v3 (DeepSeek-V3), r1 (DeepSeek-R1), gpt4o (GPT-4o)"
    )
    
    parser.add_argument(
        "--generate-data",
        action="store_true",
        default=False,
        help="Force generate new dataset (default: use existing if available)"
    )
    
    # Local model support
    parser.add_argument(
        "--solver-provider",
        type=str,
        default=None,
        choices=["siliconflow", "deepseek", "openai", "local"],
        help="Provider for solver model (default: siliconflow)"
    )
    
    parser.add_argument(
        "--solver-model",
        type=str,
        default=None,
        help="Model name for solver (e.g., /mnt/data/zql/model/llama/Llama-2-13b-chat-hf)"
    )
    
    parser.add_argument(
        "--solver-base-url",
        type=str,
        default=None,
        help="Base URL for local model server (e.g., http://localhost:8000/v1)"
    )
    
    parser.add_argument(
        "--random-sample",
        action="store_true",
        default=False,
        help="Randomly sample from existing dataset instead of using all samples"
    )
    
    args = parser.parse_args()
    
    # 验证配置
    if not config.validate():
        print("\n❌ Configuration validation failed. Please check your API keys.")
        sys.exit(1)
    
    # 运行实验
    if args.mode == "quick_test":
        asyncio.run(quick_test(
            num_samples=args.samples,
            methods=args.methods,
            generate_data=args.generate_data,
            data_file=args.data_file,
            solver_provider=args.solver_provider,
            solver_model=args.solver_model,
            solver_base_url=args.solver_base_url,
            random_sample=args.random_sample,
        ))
    elif args.mode == "generate":
        asyncio.run(generate_only(num_samples=args.samples))
    elif args.mode == "full":
        asyncio.run(full_experiment(
            num_samples=args.samples,
            methods=args.methods,
            generate_data=args.generate_data,
            data_file=args.data_file,
            solver_provider=args.solver_provider,
            solver_model=args.solver_model,
            solver_base_url=args.solver_base_url,
        ))
    elif args.mode == "run":
        if not args.data_file:
            print("❌ --data-file is required for 'run' mode")
            sys.exit(1)
        asyncio.run(run_on_dataset(args.data_file, args.methods))
    elif args.mode == "evaluate":
        if not args.data_file:
            print("❌ --data-file is required for 'evaluate' mode")
            sys.exit(1)
        if not args.responses_dir:
            print("❌ --responses-dir is required for 'evaluate' mode")
            sys.exit(1)
        asyncio.run(evaluate_only(args.data_file, args.responses_dir, args.judge))


if __name__ == "__main__":
    main()
