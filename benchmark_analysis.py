"""
基准测试结果分析与可视化模块 (benchmark_analysis.py)
生成论文级别的图表和分析报告

用于ACL论文: Bridging the Intent Gap

功能:
1. 加载实验结果
2. 生成对比图表
3. 统计显著性检验
4. Case Study分析
5. LaTeX表格生成
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠ matplotlib/seaborn not installed. Visualization disabled.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ========== 样式配置 ==========
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}

# ACL论文配色方案
COLORS = {
    "Standard": "#7f8c8d",      # 灰色
    "CoT": "#3498db",           # 蓝色
    "CoT-SC": "#9b59b6",        # 紫色
    "IARO-Base": "#e67e22",     # 橙色
    "IARO-Augment": "#f39c12",  # 黄橙
    "IARO-Hybrid": "#e74c3c",   # 红色 (主推方法)
}

METHOD_LABELS = {
    "Standard": "Standard",
    "CoT": "CoT",
    "CoT-SC": "CoT-SC",
    "IARO-Base": "IARO-Base",
    "IARO-Augment": "IARO-Aug",
    "IARO-Hybrid": "IARO-Hyb (Ours)",
}

DATASET_LABELS = {
    "sst5": "SST-5",
    "race": "RACE",
    "medmcqa": "MedMCQA",
}


# ========== 结果加载器 ==========
class ResultsLoader:
    """加载和管理实验结果"""
    
    def __init__(self, results_dir: str = "results/benchmark"):
        self.results_dir = Path(results_dir)
        self.results: Dict[str, Any] = {}
    
    def load_latest(self, dataset: str) -> Optional[Dict]:
        """加载指定数据集的最新结果"""
        pattern = f"{dataset}_*.json"
        files = sorted(self.results_dir.glob(pattern), reverse=True)
        
        if not files:
            print(f"⚠ No results found for {dataset}")
            return None
        
        latest_file = files[0]
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.results[dataset] = data
        print(f"✓ Loaded {dataset} results from {latest_file.name}")
        return data
    
    def load_all(self) -> Dict[str, Dict]:
        """加载所有数据集的最新结果"""
        for dataset in ["sst5", "race", "medmcqa"]:
            self.load_latest(dataset)
        return self.results
    
    def load_file(self, filepath: str) -> Dict:
        """加载指定文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = data.get("dataset", "unknown")
        self.results[dataset] = data
        return data


# ========== 可视化器 ==========
class BenchmarkVisualizer:
    """基准测试可视化器"""
    
    def __init__(self, output_dir: str = "figures"):
        if not HAS_PLOTTING:
            raise RuntimeError("matplotlib/seaborn not installed")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except:
            plt.style.use("default")
        
        plt.rcParams.update(STYLE_CONFIG)
    
    def plot_single_dataset_comparison(
        self,
        results: Dict,
        save_name: str = None,
    ) -> plt.Figure:
        """
        绘制单个数据集的方法对比图
        
        Args:
            results: 实验结果
            save_name: 保存文件名
        """
        dataset = results.get("dataset", "unknown")
        method_results = results.get("method_results", {})
        
        if not method_results:
            print("⚠ No method results found")
            return None
        
        # 准备数据
        methods = list(method_results.keys())
        accuracies = [method_results[m].get("accuracy", 0) for m in methods]
        
        # 排序
        sorted_indices = np.argsort(accuracies)[::-1]
        methods = [methods[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [COLORS.get(m, "#95a5a6") for m in methods]
        labels = [METHOD_LABELS.get(m, m) for m in methods]
        
        bars = ax.bar(labels, accuracies, color=colors, edgecolor='white', linewidth=1)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax.annotate(
                f'{acc:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
            )
        
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Method Comparison on {DATASET_LABELS.get(dataset, dataset)}')
        ax.set_ylim(0, 1.1)
        
        # 高亮IARO方法
        for i, method in enumerate(methods):
            if 'iaro' in method.lower():
                bars[i].set_edgecolor('#c0392b')
                bars[i].set_linewidth(2)
        
        plt.tight_layout()
        
        # 保存
        if save_name is None:
            save_name = f"{dataset}_comparison.pdf"
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_cross_dataset_comparison(
        self,
        all_results: Dict[str, Dict],
        save_name: str = "cross_dataset_comparison.pdf",
    ) -> plt.Figure:
        """
        绘制跨数据集方法对比图
        
        Args:
            all_results: 所有数据集的结果
            save_name: 保存文件名
        """
        if not all_results:
            print("⚠ No results to plot")
            return None
        
        # 收集所有方法
        all_methods = set()
        for results in all_results.values():
            all_methods.update(results.get("method_results", {}).keys())
        
        methods = sorted(all_methods, key=lambda m: ('iaro' not in m.lower(), m))
        datasets = list(all_results.keys())
        
        # 准备数据
        data = []
        for dataset in datasets:
            method_results = all_results[dataset].get("method_results", {})
            for method in methods:
                acc = method_results.get(method, {}).get("accuracy", 0)
                data.append({
                    "Dataset": DATASET_LABELS.get(dataset, dataset),
                    "Method": METHOD_LABELS.get(method, method),
                    "Accuracy": acc,
                    "method_key": method,
                })
        
        if HAS_PANDAS:
            df = pd.DataFrame(data)
        else:
            # 手动处理
            pass
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_datasets = len(datasets)
        n_methods = len(methods)
        bar_width = 0.8 / n_methods
        
        x = np.arange(n_datasets)
        
        for i, method in enumerate(methods):
            values = []
            for dataset in datasets:
                method_results = all_results[dataset].get("method_results", {})
                acc = method_results.get(method, {}).get("accuracy", 0)
                values.append(acc)
            
            offset = (i - n_methods / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                values,
                bar_width,
                label=METHOD_LABELS.get(method, method),
                color=COLORS.get(method, f"C{i}"),
            )
            
            # IARO方法突出显示
            if 'iaro' in method.lower() and 'hybrid' in method.lower():
                for bar, val in zip(bars, values):
                    ax.annotate(
                        f'{val:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                    )
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Cross-Dataset Method Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
        ax.legend(loc='upper right', ncol=2)
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_difficulty_analysis(
        self,
        results: Dict,
        save_name: str = None,
    ) -> plt.Figure:
        """
        绘制按难度分层的性能分析图
        
        Args:
            results: 实验结果
            save_name: 保存文件名
        """
        dataset = results.get("dataset", "unknown")
        method_results = results.get("method_results", {})
        
        # 收集难度数据
        difficulties = ["easy", "medium", "hard"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(difficulties))
        bar_width = 0.8 / len(method_results)
        
        for i, (method, data) in enumerate(method_results.items()):
            diff_acc = data.get("difficulty_accuracy", {})
            values = [diff_acc.get(d, 0) for d in difficulties]
            
            offset = (i - len(method_results) / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                values,
                bar_width,
                label=METHOD_LABELS.get(method, method),
                color=COLORS.get(method, f"C{i}"),
            )
        
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Performance by Difficulty on {DATASET_LABELS.get(dataset, dataset)}')
        ax.set_xticks(x)
        ax.set_xticklabels(['Easy', 'Medium', 'Hard'])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f"{dataset}_difficulty_analysis.pdf"
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_improvement_chart(
        self,
        all_results: Dict[str, Dict],
        baseline: str = "Standard",
        save_name: str = "iaro_improvement.pdf",
    ) -> plt.Figure:
        """
        绘制IARO相对于baseline的提升图
        
        Args:
            all_results: 所有数据集结果
            baseline: 基线方法
            save_name: 保存文件名
        """
        datasets = list(all_results.keys())
        
        # 找最佳IARO方法
        improvements = []
        for dataset in datasets:
            method_results = all_results[dataset].get("method_results", {})
            
            baseline_acc = method_results.get(baseline, {}).get("accuracy", 0)
            
            # 找最佳IARO
            iaro_methods = [m for m in method_results.keys() if 'iaro' in m.lower()]
            if iaro_methods and baseline_acc > 0:
                best_iaro = max(iaro_methods, key=lambda m: method_results[m].get("accuracy", 0))
                iaro_acc = method_results[best_iaro].get("accuracy", 0)
                improvement = (iaro_acc - baseline_acc) / baseline_acc * 100
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(8, 5))
        
        colors = ['#e74c3c' if imp > 0 else '#95a5a6' for imp in improvements]
        labels = [DATASET_LABELS.get(d, d) for d in datasets]
        
        bars = ax.bar(labels, improvements, color=colors)
        
        # 添加数值标签
        for bar, val in zip(bars, improvements):
            label = f'+{val:.1f}%' if val > 0 else f'{val:.1f}%'
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 if val >= 0 else -15),
                textcoords="offset points",
                ha='center',
                va='bottom' if val >= 0 else 'top',
                fontsize=11,
                fontweight='bold',
            )
        
        ax.set_ylabel(f'Improvement over {baseline} (%)')
        ax.set_title('IARO Improvement Across Datasets')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_efficiency_comparison(
        self,
        all_results: Dict[str, Dict],
        save_name: str = "efficiency_comparison.pdf",
    ) -> plt.Figure:
        """
        绘制效率对比图 (Accuracy vs API Calls)
        
        Args:
            all_results: 所有数据集结果
            save_name: 保存文件名
        """
        # 汇总所有数据集的平均性能
        method_avg = defaultdict(lambda: {"accuracy": [], "api_calls": []})
        
        for dataset, results in all_results.items():
            method_results = results.get("method_results", {})
            for method, data in method_results.items():
                method_avg[method]["accuracy"].append(data.get("accuracy", 0))
                method_avg[method]["api_calls"].append(data.get("api_calls", 1))
        
        # 计算平均
        methods = list(method_avg.keys())
        avg_acc = [np.mean(method_avg[m]["accuracy"]) for m in methods]
        avg_calls = [np.mean(method_avg[m]["api_calls"]) for m in methods]
        
        # 计算效率 (accuracy per call)
        efficiency = [a / c if c > 0 else 0 for a, c in zip(avg_acc, avg_calls)]
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: Accuracy vs API Calls
        colors = [COLORS.get(m, "#95a5a6") for m in methods]
        
        for i, method in enumerate(methods):
            marker = '*' if 'iaro' in method.lower() else 'o'
            size = 200 if 'iaro' in method.lower() else 100
            ax1.scatter(
                avg_calls[i], avg_acc[i],
                s=size,
                c=colors[i],
                marker=marker,
                label=METHOD_LABELS.get(method, method),
                edgecolors='white',
                linewidth=1,
            )
        
        ax1.set_xlabel('Average API Calls')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Accuracy vs. Computational Cost')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 右图: Efficiency Bar
        sorted_indices = np.argsort(efficiency)[::-1]
        sorted_methods = [methods[i] for i in sorted_indices]
        sorted_efficiency = [efficiency[i] for i in sorted_indices]
        sorted_colors = [COLORS.get(m, "#95a5a6") for m in sorted_methods]
        
        labels = [METHOD_LABELS.get(m, m) for m in sorted_methods]
        
        ax2.barh(labels, sorted_efficiency, color=sorted_colors)
        ax2.set_xlabel('Efficiency (Accuracy / API Calls)')
        ax2.set_title('Method Efficiency Ranking')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_api_cost_comparison(
        self,
        all_results: Dict[str, Dict],
        save_name: str = "api_cost_comparison.pdf",
    ) -> plt.Figure:
        """
        绘制API成本对比图
        
        Args:
            all_results: 所有数据集结果
            save_name: 保存文件名
        """
        # 汇总所有数据集的API调用
        method_calls = defaultdict(list)
        
        for dataset, results in all_results.items():
            method_results = results.get("method_results", {})
            for method, data in method_results.items():
                method_calls[method].append(data.get("api_calls", 0))
        
        methods = list(method_calls.keys())
        avg_calls = [np.mean(method_calls[m]) for m in methods]
        
        # 估算成本 (USD)
        API_COST_PER_1K = {"input": 0.001, "output": 0.002}
        est_costs = [c * (500 * API_COST_PER_1K["input"] + 200 * API_COST_PER_1K["output"]) / 1000 for c in avg_calls]
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = [COLORS.get(m, "#95a5a6") for m in methods]
        labels = [METHOD_LABELS.get(m, m) for m in methods]
        
        # 左图: API调用次数
        sorted_idx = np.argsort(avg_calls)
        sorted_methods = [methods[i] for i in sorted_idx]
        sorted_calls = [avg_calls[i] for i in sorted_idx]
        sorted_colors = [COLORS.get(m, "#95a5a6") for m in sorted_methods]
        sorted_labels = [METHOD_LABELS.get(m, m) for m in sorted_methods]
        
        bars1 = ax1.barh(sorted_labels, sorted_calls, color=sorted_colors)
        ax1.set_xlabel('Average API Calls per Dataset')
        ax1.set_title('API Call Comparison')
        
        # 标注数值
        for bar, val in zip(bars1, sorted_calls):
            ax1.annotate(
                f'{val:.0f}',
                xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                xytext=(3, 0),
                textcoords="offset points",
                ha='left',
                va='center',
                fontsize=9,
            )
        
        # 右图: 估算成本
        sorted_costs = [est_costs[i] for i in sorted_idx]
        bars2 = ax2.barh(sorted_labels, sorted_costs, color=sorted_colors)
        ax2.set_xlabel('Estimated Cost (USD)')
        ax2.set_title('API Cost Comparison')
        
        for bar, val in zip(bars2, sorted_costs):
            ax2.annotate(
                f'${val:.4f}',
                xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                xytext=(3, 0),
                textcoords="offset points",
                ha='left',
                va='center',
                fontsize=9,
            )
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_accuracy_vs_cost(
        self,
        all_results: Dict[str, Dict],
        save_name: str = "accuracy_vs_cost.pdf",
    ) -> plt.Figure:
        """
        绘制准确率vs成本散点图
        
        Args:
            all_results: 所有数据集结果
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 汇总所有数据集
        method_stats = defaultdict(lambda: {"accuracy": [], "api_calls": []})
        
        for dataset, results in all_results.items():
            method_results = results.get("method_results", {})
            for method, data in method_results.items():
                method_stats[method]["accuracy"].append(data.get("accuracy", 0))
                method_stats[method]["api_calls"].append(data.get("api_calls", 0))
        
        methods = list(method_stats.keys())
        
        for method in methods:
            avg_acc = np.mean(method_stats[method]["accuracy"])
            avg_calls = np.mean(method_stats[method]["api_calls"])
            
            is_iaro = 'iaro' in method.lower()
            marker = '*' if is_iaro else 'o'
            size = 300 if is_iaro else 150
            color = COLORS.get(method, "#95a5a6")
            
            ax.scatter(
                avg_calls, avg_acc,
                s=size,
                c=color,
                marker=marker,
                label=METHOD_LABELS.get(method, method),
                edgecolors='white' if not is_iaro else 'black',
                linewidth=2 if is_iaro else 1,
                zorder=10 if is_iaro else 5,
            )
            
            # 标注方法名
            ax.annotate(
                METHOD_LABELS.get(method, method),
                xy=(avg_calls, avg_acc),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight='bold' if is_iaro else 'normal',
            )
        
        ax.set_xlabel('Average API Calls', fontsize=12)
        ax.set_ylabel('Average Accuracy', fontsize=12)
        ax.set_title('Accuracy vs. API Cost Trade-off', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加Pareto前沿线提示
        ax.annotate(
            '← Lower Cost',
            xy=(0.02, 0.02),
            xycoords='axes fraction',
            fontsize=10,
            color='gray',
        )
        ax.annotate(
            'Higher Accuracy →',
            xy=(0.7, 0.98),
            xycoords='axes fraction',
            fontsize=10,
            color='gray',
        )
        
        ax.legend(loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {save_path}")
        
        return fig
    
    def generate_all_figures(self, all_results: Dict[str, Dict]):
        """生成所有图表"""
        print("\n" + "=" * 60)
        print("Generating All Figures")
        print("=" * 60)
        
        # 单数据集图表
        for dataset, results in all_results.items():
            self.plot_single_dataset_comparison(results)
            self.plot_difficulty_analysis(results)
        
        # 跨数据集图表
        if len(all_results) > 1:
            self.plot_cross_dataset_comparison(all_results)
            self.plot_improvement_chart(all_results)
            self.plot_efficiency_comparison(all_results)
            self.plot_api_cost_comparison(all_results)
            self.plot_accuracy_vs_cost(all_results)
        
        print(f"\n✓ All figures saved to {self.output_dir}")


# ========== LaTeX表格生成器 ==========
class LaTeXTableGenerator:
    """生成LaTeX格式的表格"""
    
    @staticmethod
    def generate_main_results_table(
        all_results: Dict[str, Dict],
        output_file: str = None,
    ) -> str:
        """
        生成主结果表格
        
        Args:
            all_results: 所有数据集结果
            output_file: 输出文件路径
        
        Returns:
            LaTeX表格字符串
        """
        datasets = list(all_results.keys())
        
        # 收集所有方法
        all_methods = set()
        for results in all_results.values():
            all_methods.update(results.get("method_results", {}).keys())
        
        methods = sorted(all_methods, key=lambda m: ('iaro' not in m.lower(), m))
        
        # 生成LaTeX
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        
        # 表头
        cols = "l" + "c" * len(datasets)
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        
        header = "Method & " + " & ".join([DATASET_LABELS.get(d, d) for d in datasets]) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        # 数据行
        best_per_dataset = {}
        for dataset in datasets:
            method_results = all_results[dataset].get("method_results", {})
            if method_results:
                best_method = max(method_results.keys(), key=lambda m: method_results[m].get("accuracy", 0))
                best_per_dataset[dataset] = best_method
        
        for method in methods:
            row_data = [METHOD_LABELS.get(method, method)]
            
            for dataset in datasets:
                method_results = all_results[dataset].get("method_results", {})
                acc = method_results.get(method, {}).get("accuracy", 0)
                
                # 最佳方法加粗
                if best_per_dataset.get(dataset) == method:
                    row_data.append(f"\\textbf{{{acc:.1%}}}")
                else:
                    row_data.append(f"{acc:.1%}")
            
            # IARO方法用不同格式
            if 'iaro' in method.lower():
                lines.append(" & ".join(row_data) + " \\\\")
            else:
                lines.append(" & ".join(row_data) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Main results on benchmark datasets. Best results are in \\textbf{bold}.}")
        lines.append("\\label{tab:main_results}")
        lines.append("\\end{table}")
        
        table_str = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(table_str)
            print(f"✓ LaTeX table saved to {output_file}")
        
        return table_str
    
    @staticmethod
    def generate_improvement_table(
        all_results: Dict[str, Dict],
        baseline: str = "Standard",
        output_file: str = None,
    ) -> str:
        """生成提升对比表格"""
        datasets = list(all_results.keys())
        
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("Dataset & Best Baseline & IARO-Hybrid & Improvement \\\\")
        lines.append("\\midrule")
        
        for dataset in datasets:
            method_results = all_results[dataset].get("method_results", {})
            
            # 找最佳baseline
            baselines = [m for m in method_results.keys() if 'iaro' not in m.lower()]
            if baselines:
                best_baseline = max(baselines, key=lambda m: method_results[m].get("accuracy", 0))
                base_acc = method_results[best_baseline].get("accuracy", 0)
            else:
                best_baseline = baseline
                base_acc = method_results.get(baseline, {}).get("accuracy", 0)
            
            # 找IARO-Hybrid
            iaro_acc = method_results.get("IARO-Hybrid", {}).get("accuracy", 0)
            
            improvement = (iaro_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0
            
            lines.append(
                f"{DATASET_LABELS.get(dataset, dataset)} & "
                f"{best_baseline} ({base_acc:.1%}) & "
                f"{iaro_acc:.1%} & "
                f"+{improvement:.1f}\\% \\\\"
            )
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{IARO improvement over best baseline.}")
        lines.append("\\label{tab:improvement}")
        lines.append("\\end{table}")
        
        table_str = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(table_str)
            print(f"✓ LaTeX table saved to {output_file}")
        
        return table_str


# ========== Case Study分析器 ==========
class CaseStudyAnalyzer:
    """Case Study分析器"""
    
    @staticmethod
    def find_representative_cases(
        results: Dict,
        num_cases: int = 5,
    ) -> List[Dict]:
        """
        找出代表性的Case Study样本
        
        Args:
            results: 实验结果
            num_cases: 案例数量
        
        Returns:
            案例列表
        """
        sample_details = results.get("sample_details", [])
        
        if not sample_details:
            print("⚠ No sample details found for case study")
            return []
        
        # 按难度分组
        cases_by_difficulty = defaultdict(list)
        for case in sample_details:
            diff = case.get("difficulty", "unknown")
            cases_by_difficulty[diff].append(case)
        
        # 从每个难度选取
        selected = []
        for diff in ["hard", "medium", "easy"]:
            cases = cases_by_difficulty.get(diff, [])
            if cases:
                selected.extend(cases[:max(1, num_cases // 3)])
        
        return selected[:num_cases]
    
    @staticmethod
    def format_case_study(cases: List[Dict], output_file: str = None) -> str:
        """格式化Case Study输出"""
        lines = []
        lines.append("# Case Study Analysis\n")
        lines.append("The following cases demonstrate IARO's advantage over baseline methods.\n")
        
        for i, case in enumerate(cases, 1):
            lines.append(f"## Case {i} (Difficulty: {case.get('difficulty', 'unknown')})\n")
            lines.append(f"**Text**: {case.get('text', 'N/A')}\n")
            
            if case.get('context'):
                lines.append(f"**Context**: {case.get('context')[:200]}...\n")
            
            if case.get('options'):
                lines.append("**Options**:")
                for j, opt in enumerate(case['options']):
                    lines.append(f"  - {chr(65+j)}. {opt}")
                lines.append("")
            
            lines.append(f"**Ground Truth**: {case.get('ground_truth', 'N/A')}")
            lines.append(f"**IARO Prediction**: {case.get('iaro_prediction', 'N/A')}")
            
            if case.get('other_predictions'):
                lines.append("**Baseline Predictions**:")
                for method, pred in case['other_predictions'].items():
                    lines.append(f"  - {method}: {pred}")
            
            lines.append("\n---\n")
        
        content = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Case study saved to {output_file}")
        
        return content


# ========== 主分析器 ==========
class BenchmarkAnalyzer:
    """基准测试综合分析器"""
    
    def __init__(self, results_dir: str = "results/benchmark", output_dir: str = "figures"):
        self.loader = ResultsLoader(results_dir)
        self.visualizer = BenchmarkVisualizer(output_dir) if HAS_PLOTTING else None
        self.latex_gen = LaTeXTableGenerator()
        self.case_analyzer = CaseStudyAnalyzer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("\n" + "=" * 60)
        print("Running Full Benchmark Analysis")
        print("=" * 60)
        
        # 加载结果
        all_results = self.loader.load_all()
        
        if not all_results:
            print("⚠ No results found to analyze")
            return
        
        # 生成图表
        if self.visualizer:
            self.visualizer.generate_all_figures(all_results)
        
        # 生成LaTeX表格
        self.latex_gen.generate_main_results_table(
            all_results,
            output_file=str(self.output_dir / "main_results.tex")
        )
        
        self.latex_gen.generate_improvement_table(
            all_results,
            output_file=str(self.output_dir / "improvement.tex")
        )
        
        # Case Study
        for dataset, results in all_results.items():
            cases = self.case_analyzer.find_representative_cases(results)
            if cases:
                self.case_analyzer.format_case_study(
                    cases,
                    output_file=str(self.output_dir / f"{dataset}_case_study.md")
                )
        
        print("\n" + "=" * 60)
        print("Analysis Complete")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='Benchmark Analysis and Visualization')
    parser.add_argument('--results-dir', type=str, default='results/benchmark',
                        help='Results directory')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Analyze specific dataset only')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
    
    if args.dataset:
        results = analyzer.loader.load_latest(args.dataset)
        if results and analyzer.visualizer:
            analyzer.visualizer.plot_single_dataset_comparison(results)
            analyzer.visualizer.plot_difficulty_analysis(results)
    else:
        analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
