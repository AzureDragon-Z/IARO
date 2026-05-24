"""
基准测试评估器模块 (benchmark_evaluator.py)
为分类和多选题任务提供评估指标计算

用于ACL论文: Bridging the Intent Gap

核心指标:
- Accuracy: 准确率
- Macro-F1: 宏平均F1
- Class-wise Accuracy: 各类别准确率
- Difficulty-wise Analysis: 按难度分层分析
"""

import json
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from benchmark_adapter import BenchmarkSample


# ========== 评估结果数据结构 ==========
@dataclass
class MethodResult:
    """单个方法的评估结果"""
    method: str
    dataset: str
    
    # 核心指标
    accuracy: float = 0.0
    macro_f1: float = 0.0
    
    # 详细指标
    class_accuracy: Dict[str, float] = field(default_factory=dict)
    class_f1: Dict[str, float] = field(default_factory=dict)
    difficulty_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # 预测详情
    predictions: List[str] = field(default_factory=list)
    ground_truth: List[str] = field(default_factory=list)
    correct_mask: List[bool] = field(default_factory=list)
    
    # 统计信息
    total_samples: int = 0
    correct_samples: int = 0
    api_calls: int = 0
    
    # 元数据
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResults:
    """完整基准测试结果"""
    experiment_id: str
    dataset: str
    num_samples: int
    
    # 各方法结果
    method_results: Dict[str, MethodResult] = field(default_factory=dict)
    
    # 样本级别详情
    sample_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # 对比分析
    comparison: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["method_results"] = {k: v.to_dict() for k, v in self.method_results.items()}
        return result
    
    def save(self, filepath: str):
        """保存结果到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"✓ Results saved to {filepath}")


class NumpyEncoder(json.JSONEncoder):
    """JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ========== 评估器类 ==========
class BenchmarkEvaluator:
    """基准测试评估器"""
    
    def __init__(self, dataset: str, label_list: List[str] = None):
        """
        初始化评估器
        
        Args:
            dataset: 数据集名称
            label_list: 标签列表
        """
        self.dataset = dataset
        self.label_list = label_list or []
    
    def evaluate_method(
        self,
        predictions: List[str],
        samples: List[BenchmarkSample],
        method_name: str,
        api_calls: int = 0,
    ) -> MethodResult:
        """
        评估单个方法的性能
        
        Args:
            predictions: 预测结果列表
            samples: 样本列表
            method_name: 方法名称
            api_calls: API调用次数
        
        Returns:
            MethodResult
        """
        ground_truth = [s.label for s in samples]
        
        # 确保预测和真实标签长度一致
        assert len(predictions) == len(ground_truth), \
            f"Prediction length ({len(predictions)}) != ground truth length ({len(ground_truth)})"
        
        # 计算正确性
        correct_mask = [p == g for p, g in zip(predictions, ground_truth)]
        correct_count = sum(correct_mask)
        total_count = len(predictions)
        
        # 核心指标
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # 各类别准确率
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for pred, gt, correct in zip(predictions, ground_truth, correct_mask):
            class_total[gt] += 1
            if correct:
                class_correct[gt] += 1
        
        class_accuracy = {
            cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
            for cls in class_total.keys()
        }
        
        # 按难度分层准确率
        difficulty_correct = defaultdict(int)
        difficulty_total = defaultdict(int)
        
        for sample, correct in zip(samples, correct_mask):
            diff = sample.difficulty
            difficulty_total[diff] += 1
            if correct:
                difficulty_correct[diff] += 1
        
        difficulty_accuracy = {
            diff: difficulty_correct[diff] / difficulty_total[diff] if difficulty_total[diff] > 0 else 0.0
            for diff in difficulty_total.keys()
        }
        
        # 计算Macro-F1
        macro_f1 = self._calculate_macro_f1(predictions, ground_truth)
        class_f1 = self._calculate_class_f1(predictions, ground_truth)
        
        return MethodResult(
            method=method_name,
            dataset=self.dataset,
            accuracy=accuracy,
            macro_f1=macro_f1,
            class_accuracy=class_accuracy,
            class_f1=class_f1,
            difficulty_accuracy=difficulty_accuracy,
            predictions=predictions,
            ground_truth=ground_truth,
            correct_mask=correct_mask,
            total_samples=total_count,
            correct_samples=correct_count,
            api_calls=api_calls,
            timestamp=datetime.now().isoformat(),
        )
    
    def _calculate_macro_f1(self, predictions: List[str], ground_truth: List[str]) -> float:
        """计算Macro-F1"""
        classes = set(ground_truth)
        f1_scores = []
        
        for cls in classes:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    def _calculate_class_f1(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """计算各类别F1"""
        classes = set(ground_truth)
        class_f1 = {}
        
        for cls in classes:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            class_f1[cls] = f1
        
        return class_f1
    
    def compare_methods(
        self,
        method_results: Dict[str, MethodResult],
        baseline: str = "Standard",
    ) -> Dict[str, Any]:
        """
        对比多个方法的性能
        
        Args:
            method_results: 方法名到结果的映射
            baseline: 基线方法名称
        
        Returns:
            对比分析结果
        """
        if baseline not in method_results:
            baseline = list(method_results.keys())[0] if method_results else None
        
        comparison = {
            "baseline": baseline,
            "methods": {},
            "rankings": {},
            "iaro_advantage": {},
        }
        
        if not baseline or not method_results:
            return comparison
        
        baseline_result = method_results[baseline]
        baseline_acc = baseline_result.accuracy
        
        # 计算各方法相对提升
        for method, result in method_results.items():
            improvement = ((result.accuracy - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
            comparison["methods"][method] = {
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "improvement_over_baseline": improvement,
                "api_calls": result.api_calls,
                "efficiency": result.accuracy / result.api_calls if result.api_calls > 0 else 0,
            }
        
        # 排名
        sorted_methods = sorted(method_results.items(), key=lambda x: x[1].accuracy, reverse=True)
        comparison["rankings"] = {method: rank + 1 for rank, (method, _) in enumerate(sorted_methods)}
        
        # IARO优势分析 (按难度)
        iaro_methods = [m for m in method_results.keys() if 'iaro' in m.lower()]
        if iaro_methods and baseline:
            iaro_method = iaro_methods[0]
            iaro_result = method_results[iaro_method]
            
            for diff in iaro_result.difficulty_accuracy.keys():
                iaro_acc = iaro_result.difficulty_accuracy.get(diff, 0)
                base_acc = baseline_result.difficulty_accuracy.get(diff, 0)
                comparison["iaro_advantage"][diff] = {
                    "iaro": iaro_acc,
                    "baseline": base_acc,
                    "gain": iaro_acc - base_acc,
                }
        
        return comparison
    
    def find_iaro_wins(
        self,
        samples: List[BenchmarkSample],
        method_results: Dict[str, MethodResult],
        iaro_method: str = None,
    ) -> List[Dict[str, Any]]:
        """
        找出IARO方法相对于所有baseline胜出的样本
        
        Args:
            samples: 样本列表
            method_results: 方法结果
            iaro_method: IARO方法名称
        
        Returns:
            IARO胜出的样本详情列表
        """
        if iaro_method is None:
            iaro_methods = [m for m in method_results.keys() if 'iaro' in m.lower()]
            iaro_method = iaro_methods[0] if iaro_methods else None
        
        if iaro_method is None or iaro_method not in method_results:
            return []
        
        iaro_result = method_results[iaro_method]
        other_methods = [m for m in method_results.keys() if m != iaro_method]
        
        wins = []
        
        for i, sample in enumerate(samples):
            iaro_correct = iaro_result.correct_mask[i]
            
            if iaro_correct:
                # 检查是否所有其他方法都错误
                all_others_wrong = all(
                    not method_results[m].correct_mask[i] for m in other_methods
                )
                
                if all_others_wrong:
                    wins.append({
                        "sample_id": sample.id,
                        "text": sample.text[:200],
                        "context": sample.context[:200] if sample.context else "",
                        "options": sample.options,
                        "ground_truth": sample.label,
                        "iaro_prediction": iaro_result.predictions[i],
                        "other_predictions": {m: method_results[m].predictions[i] for m in other_methods},
                        "difficulty": sample.difficulty,
                        "subdomain": sample.subdomain,
                    })
        
        return wins
    
    def generate_confusion_matrix(
        self,
        predictions: List[str],
        ground_truth: List[str],
    ) -> Dict[str, Dict[str, int]]:
        """生成混淆矩阵"""
        labels = sorted(set(ground_truth))
        matrix = {label: {l: 0 for l in labels} for label in labels}
        
        for pred, gt in zip(predictions, ground_truth):
            if gt in matrix and pred in matrix[gt]:
                matrix[gt][pred] += 1
        
        return matrix


# ========== 统计检验 ==========
class StatisticalTests:
    """统计显著性检验"""
    
    @staticmethod
    def paired_bootstrap_test(
        scores_a: List[float],
        scores_b: List[float],
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """
        配对Bootstrap显著性检验
        
        Args:
            scores_a: 方法A的得分
            scores_b: 方法B的得分
            n_bootstrap: Bootstrap次数
            seed: 随机种子
        
        Returns:
            (mean_diff, p_value)
        """
        np.random.seed(seed)
        
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        n = len(scores_a)
        
        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        
        # Bootstrap
        count = 0
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, n, n)
            boot_diff = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
            if boot_diff <= 0:
                count += 1
        
        p_value = count / n_bootstrap
        
        return observed_diff, p_value
    
    @staticmethod
    def mcnemar_test(
        correct_a: List[bool],
        correct_b: List[bool],
    ) -> Tuple[float, float]:
        """
        McNemar检验 (用于比较两个分类器)
        
        Args:
            correct_a: 方法A的正确性列表
            correct_b: 方法B的正确性列表
        
        Returns:
            (chi2, p_value)
        """
        # 构建2x2列联表
        # a: A对B对, b: A对B错, c: A错B对, d: A错B错
        b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
        c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
        
        # McNemar统计量
        if b + c == 0:
            return 0.0, 1.0
        
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        
        # 简化的p值计算 (使用正态近似)
        from math import erfc, sqrt
        p_value = erfc(sqrt(chi2 / 2))
        
        return chi2, p_value


# ========== 报告生成 ==========
def generate_summary_table(results: BenchmarkResults) -> str:
    """生成Markdown格式的结果汇总表"""
    lines = []
    lines.append(f"# {results.dataset.upper()} Evaluation Results\n")
    lines.append(f"**Samples**: {results.num_samples} | **Time**: {results.timestamp}\n")
    
    # 方法对比表
    lines.append("## Method Comparison\n")
    lines.append("| Method | Accuracy | Macro-F1 | API Calls | Efficiency |")
    lines.append("|--------|----------|----------|-----------|------------|")
    
    sorted_methods = sorted(
        results.method_results.items(),
        key=lambda x: x[1].accuracy,
        reverse=True
    )
    
    for method, result in sorted_methods:
        eff = result.accuracy / result.api_calls if result.api_calls > 0 else 0
        marker = " ★" if 'iaro' in method.lower() else ""
        lines.append(
            f"| {method}{marker} | {result.accuracy:.1%} | {result.macro_f1:.3f} | "
            f"{result.api_calls} | {eff:.4f} |"
        )
    
    # 最佳基线 vs IARO
    baselines = [m for m in results.method_results.keys() if 'iaro' not in m.lower()]
    iaro_methods = [m for m in results.method_results.keys() if 'iaro' in m.lower()]
    
    if baselines and iaro_methods:
        best_baseline = max(baselines, key=lambda m: results.method_results[m].accuracy)
        best_iaro = max(iaro_methods, key=lambda m: results.method_results[m].accuracy)
        
        base_acc = results.method_results[best_baseline].accuracy
        iaro_acc = results.method_results[best_iaro].accuracy
        improvement = (iaro_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0
        
        lines.append(f"\n## Key Findings")
        lines.append(f"- **Best Baseline**: {best_baseline} ({base_acc:.1%})")
        lines.append(f"- **Best IARO**: {best_iaro} ({iaro_acc:.1%})")
        lines.append(f"- **Improvement**: {improvement:+.1f}%")
    
    return "\n".join(lines)


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Testing BenchmarkEvaluator...")
    
    # 创建模拟数据
    from benchmark_adapter import BenchmarkSample
    
    samples = [
        BenchmarkSample(id="1", dataset="test", text="test1", label="A", difficulty="easy"),
        BenchmarkSample(id="2", dataset="test", text="test2", label="B", difficulty="medium"),
        BenchmarkSample(id="3", dataset="test", text="test3", label="A", difficulty="hard"),
        BenchmarkSample(id="4", dataset="test", text="test4", label="C", difficulty="easy"),
        BenchmarkSample(id="5", dataset="test", text="test5", label="A", difficulty="medium"),
    ]
    
    predictions_standard = ["A", "B", "B", "C", "B"]  # 3/5 correct
    predictions_iaro = ["A", "B", "A", "C", "A"]  # 5/5 correct
    
    evaluator = BenchmarkEvaluator("test", ["A", "B", "C"])
    
    result_standard = evaluator.evaluate_method(predictions_standard, samples, "Standard", api_calls=5)
    result_iaro = evaluator.evaluate_method(predictions_iaro, samples, "IARO", api_calls=15)
    
    print(f"Standard Accuracy: {result_standard.accuracy:.1%}")
    print(f"IARO Accuracy: {result_iaro.accuracy:.1%}")
    
    comparison = evaluator.compare_methods({"Standard": result_standard, "IARO": result_iaro})
    print(f"IARO Improvement: {comparison['methods']['IARO']['improvement_over_baseline']:.1f}%")
    
    print("\n✓ Evaluator test completed")
