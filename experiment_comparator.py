import os
import glob
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, f_oneway


@dataclass
class GroupData:
    name: str
    data: pd.DataFrame
    file_path: str
    
    @property
    def numeric_columns(self) -> List[str]:
        return self.data.select_dtypes(include=[np.number]).columns.tolist()


@dataclass
class TTestResult:
    group1: str
    group2: str
    column: str
    t_statistic: float
    p_value: float
    degrees_of_freedom: float
    mean1: float
    mean2: float
    pooled_variance: float
    is_significant: bool
    significance_level: float = 0.05


@dataclass
class ANOVAResult:
    groups: List[str]
    column: str
    f_statistic: float
    p_value: float
    degrees_of_freedom_between: int
    degrees_of_freedom_within: int
    is_significant: bool
    group_means: Dict[str, float]
    significance_level: float = 0.05
    post_hoc_tests: List[Tuple[str, str, float, bool]] = field(default_factory=list)


@dataclass
class AnalysisReport:
    timestamp: str
    groups_analyzed: List[str]
    columns_analyzed: List[str]
    t_test_results: Dict[str, Dict[str, Dict[str, TTestResult]]]
    anova_results: Dict[str, ANOVAResult]
    summary_statistics: Dict[str, pd.DataFrame]
    significance_threshold: float = 0.05


class ExperimentComparator:
    def __init__(self, significance_level: float = 0.05, paired_t_test: bool = False):
        self.significance_level = significance_level
        self.paired_t_test = paired_t_test
        self.groups: Dict[str, GroupData] = {}
        
    def load_csv_file(self, file_path: str, group_name: Optional[str] = None) -> GroupData:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_csv(file_path)
        
        if group_name is None:
            group_name = os.path.splitext(os.path.basename(file_path))[0]
        
        group_data = GroupData(
            name=group_name,
            data=data,
            file_path=file_path
        )
        
        self.groups[group_name] = group_data
        return group_data
    
    def load_multiple_csv(self, directory: str, pattern: str = "*.csv") -> List[GroupData]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = glob.glob(os.path.join(directory, pattern))
        loaded_groups = []
        
        for file_path in files:
            group_data = self.load_csv_file(file_path)
            loaded_groups.append(group_data)
        
        return loaded_groups
    
    def get_common_numeric_columns(self) -> List[str]:
        if not self.groups:
            return []
        
        common_columns = None
        for group_name, group_data in self.groups.items():
            if common_columns is None:
                common_columns = set(group_data.numeric_columns)
            else:
                common_columns = common_columns.intersection(set(group_data.numeric_columns))
        
        return sorted(list(common_columns)) if common_columns else []
    
    def perform_t_test(self, group1_name: str, group2_name: str, 
                       column: str) -> Optional[TTestResult]:
        if group1_name not in self.groups or group2_name not in self.groups:
            raise ValueError(f"Group not found. Available groups: {list(self.groups.keys())}")
        
        group1 = self.groups[group1_name]
        group2 = self.groups[group2_name]
        
        if column not in group1.data.columns or column not in group2.data.columns:
            raise ValueError(f"Column '{column}' not found in one or both groups")
        
        data1 = group1.data[column].dropna()
        data2 = group2.data[column].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            return None
        
        mean1 = data1.mean()
        mean2 = data2.mean()
        
        if self.paired_t_test:
            min_len = min(len(data1), len(data2))
            data1 = data1.iloc[:min_len]
            data2 = data2.iloc[:min_len]
            t_statistic, p_value = ttest_rel(data1, data2)
            df = min_len - 1
            pooled_variance = ((data1.var() + data2.var()) / 2)
        else:
            t_statistic, p_value = ttest_ind(data1, data2, equal_var=False)
            n1, n2 = len(data1), len(data2)
            var1, var2 = data1.var(ddof=1), data2.var(ddof=1)
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            pooled_variance = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
        
        is_significant = bool(p_value < self.significance_level)
        
        return TTestResult(
            group1=group1_name,
            group2=group2_name,
            column=column,
            t_statistic=t_statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            mean1=mean1,
            mean2=mean2,
            pooled_variance=pooled_variance,
            is_significant=is_significant,
            significance_level=self.significance_level
        )
    
    def perform_all_pairwise_t_tests(self) -> Dict[str, Dict[str, Dict[str, TTestResult]]]:
        group_names = list(self.groups.keys())
        common_columns = self.get_common_numeric_columns()
        
        results = {}
        
        for i, group1 in enumerate(group_names):
            results[group1] = {}
            for group2 in group_names[i+1:]:
                results[group1][group2] = {}
                for column in common_columns:
                    test_result = self.perform_t_test(group1, group2, column)
                    if test_result is not None:
                        results[group1][group2][column] = test_result
        
        return results
    
    def perform_anova(self, column: str) -> Optional[ANOVAResult]:
        if len(self.groups) < 3:
            raise ValueError("ANOVA requires at least 3 groups")
        
        common_columns = self.get_common_numeric_columns()
        if column not in common_columns:
            raise ValueError(f"Column '{column}' not found in all groups")
        
        group_data_list = []
        group_means = {}
        group_names = []
        
        for group_name, group_data in self.groups.items():
            data = group_data.data[column].dropna()
            if len(data) > 0:
                group_data_list.append(data.values)
                group_means[group_name] = data.mean()
                group_names.append(group_name)
        
        if len(group_data_list) < 3:
            return None
        
        f_statistic, p_value = f_oneway(*group_data_list)
        
        total_samples = sum(len(g) for g in group_data_list)
        k = len(group_data_list)
        df_between = k - 1
        df_within = total_samples - k
        
        is_significant = bool(p_value < self.significance_level)
        
        post_hoc_tests = []
        if is_significant:
            post_hoc_tests = self._perform_tukey_post_hoc(group_names, group_data_list)
        
        return ANOVAResult(
            groups=group_names,
            column=column,
            f_statistic=f_statistic,
            p_value=p_value,
            degrees_of_freedom_between=df_between,
            degrees_of_freedom_within=df_within,
            is_significant=is_significant,
            group_means=group_means,
            significance_level=self.significance_level,
            post_hoc_tests=post_hoc_tests
        )
    
    def _perform_tukey_post_hoc(self, group_names: List[str], 
                                  group_data: List[np.ndarray]) -> List[Tuple[str, str, float, bool]]:
        results = []
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                data1 = group_data[i]
                data2 = group_data[j]
                
                t_stat, p_val = ttest_ind(data1, data2, equal_var=True)
                n_groups = len(group_names)
                corrected_alpha = self.significance_level / (n_groups * (n_groups - 1) / 2)
                is_significant = bool(p_val < corrected_alpha)
                
                results.append((
                    group_names[i],
                    group_names[j],
                    p_val,
                    is_significant
                ))
        
        return results
    
    def perform_all_anova_tests(self) -> Dict[str, ANOVAResult]:
        if len(self.groups) < 3:
            return {}
        
        common_columns = self.get_common_numeric_columns()
        results = {}
        
        for column in common_columns:
            anova_result = self.perform_anova(column)
            if anova_result is not None:
                results[column] = anova_result
        
        return results
    
    def get_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        summaries = {}
        for group_name, group_data in self.groups.items():
            summaries[group_name] = group_data.data.describe()
        return summaries
    
    def get_significant_differences_summary(self, 
                                             t_test_results: Dict[str, Dict[str, Dict[str, TTestResult]]],
                                             anova_results: Dict[str, ANOVAResult]) -> Dict[str, Any]:
        significant_t_tests = []
        for group1, group2_data in t_test_results.items():
            for group2, columns in group2_data.items():
                for column, result in columns.items():
                    if result.is_significant:
                        significant_t_tests.append({
                            'group1': group1,
                            'group2': group2,
                            'column': column,
                            'p_value': result.p_value,
                            'mean_difference': result.mean1 - result.mean2
                        })
        
        significant_anova = []
        for column, result in anova_results.items():
            if result.is_significant:
                significant_anova.append({
                    'column': column,
                    'p_value': result.p_value,
                    'f_statistic': result.f_statistic,
                    'significant_post_hoc': [
                        (g1, g2, p) for g1, g2, p, sig in result.post_hoc_tests if sig
                    ]
                })
        
        return {
            'significant_t_tests': significant_t_tests,
            'significant_anova': significant_anova,
            'total_t_tests': sum(len(cols) for g1 in t_test_results for g2 in t_test_results[g1] for cols in [t_test_results[g1][g2]]),
            'total_anova_tests': len(anova_results),
            'significance_threshold': self.significance_level
        }
    
    def generate_report(self) -> AnalysisReport:
        from datetime import datetime
        
        t_test_results = self.perform_all_pairwise_t_tests()
        anova_results = self.perform_all_anova_tests()
        
        return AnalysisReport(
            timestamp=datetime.now().isoformat(),
            groups_analyzed=list(self.groups.keys()),
            columns_analyzed=self.get_common_numeric_columns(),
            t_test_results=t_test_results,
            anova_results=anova_results,
            summary_statistics=self.get_summary_statistics(),
            significance_threshold=self.significance_level
        )
    
    def export_report_to_text(self, report: AnalysisReport, 
                                output_path: Optional[str] = None) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("实验结果对比分析报告")
        lines.append("=" * 80)
        lines.append(f"\n分析时间: {report.timestamp}")
        lines.append(f"显著性水平: α = {report.significance_threshold}")
        lines.append(f"\n参与分析的组别: {', '.join(report.groups_analyzed)}")
        lines.append(f"分析的指标列: {', '.join(report.columns_analyzed)}")
        
        lines.append("\n" + "-" * 80)
        lines.append("一、描述性统计摘要")
        lines.append("-" * 80)
        
        for group_name, summary in report.summary_statistics.items():
            lines.append(f"\n【{group_name}】统计摘要:")
            lines.append(summary.to_string())
            lines.append("")
        
        lines.append("\n" + "-" * 80)
        lines.append("二、t-test 结果 (成对比较)")
        lines.append("-" * 80)
        
        for group1, group2_data in report.t_test_results.items():
            for group2, columns in group2_data.items():
                lines.append(f"\n组间比较: {group1} vs {group2}")
                for column, result in columns.items():
                    sig_marker = "***" if result.is_significant else ""
                    lines.append(f"\n  指标: {column} {sig_marker}")
                    lines.append(f"    t统计量: {result.t_statistic:.4f}")
                    lines.append(f"    p值: {result.p_value:.6f}")
                    lines.append(f"    自由度: {result.degrees_of_freedom:.2f}")
                    lines.append(f"    {group1} 均值: {result.mean1:.4f}")
                    lines.append(f"    {group2} 均值: {result.mean2:.4f}")
                    lines.append(f"    均值差: {result.mean1 - result.mean2:.4f}")
                    lines.append(f"    显著性: {'显著' if result.is_significant else '不显著'} (p < {result.significance_level})")
        
        lines.append("\n" + "-" * 80)
        lines.append("三、ANOVA 结果 (多组比较)")
        lines.append("-" * 80)
        
        if report.anova_results:
            for column, result in report.anova_results.items():
                sig_marker = "***" if result.is_significant else ""
                lines.append(f"\n指标: {column} {sig_marker}")
                lines.append(f"  F统计量: {result.f_statistic:.4f}")
                lines.append(f"  p值: {result.p_value:.6f}")
                lines.append(f"  组间自由度: {result.degrees_of_freedom_between}")
                lines.append(f"  组内自由度: {result.degrees_of_freedom_within}")
                lines.append(f"  各组均值:")
                for group_name, mean_val in result.group_means.items():
                    lines.append(f"    {group_name}: {mean_val:.4f}")
                lines.append(f"  显著性: {'显著' if result.is_significant else '不显著'} (p < {result.significance_level})")
                
                if result.is_significant and result.post_hoc_tests:
                    lines.append(f"  事后检验 (Tukey HSD 修正):")
                    for g1, g2, p_val, sig in result.post_hoc_tests:
                        sig_text = "显著" if sig else "不显著"
                        lines.append(f"    {g1} vs {g2}: p = {p_val:.6f} ({sig_text})")
        else:
            lines.append("\n  ANOVA 未执行 (需要至少3组数据)")
        
        lines.append("\n" + "-" * 80)
        lines.append("四、显著性差异汇总")
        lines.append("-" * 80)
        
        significant_differences = self.get_significant_differences_summary(
            report.t_test_results, report.anova_results
        )
        
        lines.append(f"\n总 t-test 比较次数: {significant_differences['total_t_tests']}")
        lines.append(f"显著差异的 t-test 数量: {len(significant_differences['significant_t_tests'])}")
        
        if significant_differences['significant_t_tests']:
            lines.append("\n显著差异的 t-test 详情:")
            for test in significant_differences['significant_t_tests']:
                lines.append(f"  {test['group1']} vs {test['group2']} (指标: {test['column']}):")
                lines.append(f"    p值 = {test['p_value']:.6f}, 均值差 = {test['mean_difference']:.4f}")
        
        lines.append(f"\n总 ANOVA 测试数量: {significant_differences['total_anova_tests']}")
        lines.append(f"显著差异的 ANOVA 数量: {len(significant_differences['significant_anova'])}")
        
        if significant_differences['significant_anova']:
            lines.append("\n显著差异的 ANOVA 详情:")
            for test in significant_differences['significant_anova']:
                lines.append(f"  指标: {test['column']}: F = {test['f_statistic']:.4f}, p = {test['p_value']:.6f}")
                if test['significant_post_hoc']:
                    lines.append(f"    显著事后比较: {', '.join([f'{g1}-{g2}(p={p:.4f})' for g1, g2, p in test['significant_post_hoc']])}")
        
        lines.append("\n" + "=" * 80)
        lines.append("报告结束")
        lines.append("=" * 80)
        
        report_text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def export_summary_to_csv(self, output_path: str):
        summary_data = []
        
        t_test_results = self.perform_all_pairwise_t_tests()
        for group1, group2_data in t_test_results.items():
            for group2, columns in group2_data.items():
                for column, result in columns.items():
                    summary_data.append({
                        'test_type': 't-test',
                        'group1': group1,
                        'group2': group2,
                        'column': column,
                        'statistic': result.t_statistic,
                        'p_value': result.p_value,
                        'df': result.degrees_of_freedom,
                        'is_significant': result.is_significant,
                        'mean1': result.mean1,
                        'mean2': result.mean2
                    })
        
        if len(self.groups) >= 3:
            anova_results = self.perform_all_anova_tests()
            for column, result in anova_results.items():
                summary_data.append({
                    'test_type': 'ANOVA',
                    'group1': '-',
                    'group2': '-',
                    'column': column,
                    'statistic': result.f_statistic,
                    'p_value': result.p_value,
                    'df': f"{result.degrees_of_freedom_between},{result.degrees_of_freedom_within}",
                    'is_significant': result.is_significant,
                    'mean1': '-',
                    'mean2': '-'
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
