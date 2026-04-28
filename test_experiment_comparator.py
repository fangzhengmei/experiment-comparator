import os
import tempfile
import shutil
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from experiment_comparator import (
    ExperimentComparator,
    GroupData,
    TTestResult,
    ANOVAResult,
    AnalysisReport
)


class TestGroupData:
    def test_numeric_columns_property(self):
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [4.5, 5.5, 6.5]
        })
        group_data = GroupData(
            name='test',
            data=data,
            file_path='/test/path.csv'
        )
        assert group_data.numeric_columns == ['col1', 'col3']


class TestExperimentComparator:
    @pytest.fixture
    def test_data_dir(self):
        return Path(__file__).parent / "test_data"
    
    @pytest.fixture
    def comparator(self):
        return ExperimentComparator(significance_level=0.05)
    
    @pytest.fixture
    def loaded_comparator(self, comparator, test_data_dir):
        comparator.load_multiple_csv(str(test_data_dir))
        return comparator
    
    def test_initialization(self):
        comp = ExperimentComparator(significance_level=0.01, paired_t_test=True)
        assert comp.significance_level == 0.01
        assert comp.paired_t_test == True
        assert comp.groups == {}
    
    def test_load_csv_file_success(self, test_data_dir):
        comp = ExperimentComparator()
        file_path = test_data_dir / "group_a.csv"
        group_data = comp.load_csv_file(str(file_path))
        
        assert group_data.name == "group_a"
        assert len(group_data.data) == 10
        assert "accuracy" in group_data.data.columns
        assert "group_a" in comp.groups
    
    def test_load_csv_file_custom_name(self, test_data_dir):
        comp = ExperimentComparator()
        file_path = test_data_dir / "group_a.csv"
        group_data = comp.load_csv_file(str(file_path), group_name="custom_group")
        
        assert group_data.name == "custom_group"
        assert "custom_group" in comp.groups
    
    def test_load_csv_file_not_found(self, comparator):
        with pytest.raises(FileNotFoundError):
            comparator.load_csv_file("/nonexistent/path.csv")
    
    def test_load_multiple_csv(self, test_data_dir):
        comp = ExperimentComparator()
        groups = comp.load_multiple_csv(str(test_data_dir))
        
        assert len(groups) == 3
        assert len(comp.groups) == 3
        group_names = [g.name for g in groups]
        assert "group_a" in group_names
        assert "group_b" in group_names
        assert "group_c" in group_names
    
    def test_get_common_numeric_columns(self, loaded_comparator):
        common_cols = loaded_comparator.get_common_numeric_columns()
        
        assert common_cols == ['accuracy', 'latency', 'memory_usage', 'throughput']
    
    def test_get_common_numeric_columns_empty(self, comparator):
        assert comparator.get_common_numeric_columns() == []
    
    def test_perform_t_test(self, loaded_comparator):
        result = loaded_comparator.perform_t_test("group_a", "group_b", "accuracy")
        
        assert isinstance(result, TTestResult)
        assert result.group1 == "group_a"
        assert result.group2 == "group_b"
        assert result.column == "accuracy"
        assert hasattr(result, 't_statistic')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'is_significant')
    
    def test_perform_t_test_group_not_found(self, comparator):
        with pytest.raises(ValueError):
            comparator.perform_t_test("nonexistent", "group_b", "accuracy")
    
    def test_perform_t_test_column_not_found(self, loaded_comparator):
        with pytest.raises(ValueError):
            loaded_comparator.perform_t_test("group_a", "group_b", "nonexistent_column")
    
    def test_perform_all_pairwise_t_tests(self, loaded_comparator):
        results = loaded_comparator.perform_all_pairwise_t_tests()
        
        assert "group_a" in results
        assert "group_b" in results["group_a"]
        assert "accuracy" in results["group_a"]["group_b"]
        
        result = results["group_a"]["group_b"]["accuracy"]
        assert isinstance(result, TTestResult)
    
    def test_perform_anova_less_than_three_groups(self, comparator):
        comparator.groups["a"] = GroupData(
            name="a",
            data=pd.DataFrame({'col': [1, 2, 3]}),
            file_path="/test.csv"
        )
        comparator.groups["b"] = GroupData(
            name="b",
            data=pd.DataFrame({'col': [4, 5, 6]}),
            file_path="/test2.csv"
        )
        
        with pytest.raises(ValueError):
            comparator.perform_anova("col")
    
    def test_perform_anova_success(self, loaded_comparator):
        result = loaded_comparator.perform_anova("accuracy")
        
        assert isinstance(result, ANOVAResult)
        assert result.column == "accuracy"
        assert hasattr(result, 'f_statistic')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'is_significant')
        assert len(result.group_means) == 3
    
    def test_perform_anova_column_not_in_all_groups(self, loaded_comparator):
        with pytest.raises(ValueError):
            loaded_comparator.perform_anova("nonexistent_column")
    
    def test_perform_all_anova_tests(self, loaded_comparator):
        results = loaded_comparator.perform_all_anova_tests()
        
        assert "accuracy" in results
        assert "latency" in results
        assert isinstance(results["accuracy"], ANOVAResult)
    
    def test_get_summary_statistics(self, loaded_comparator):
        summaries = loaded_comparator.get_summary_statistics()
        
        assert "group_a" in summaries
        assert "group_b" in summaries
        assert "group_c" in summaries
        
        for name, summary in summaries.items():
            assert isinstance(summary, pd.DataFrame)
            assert "accuracy" in summary.columns
    
    def test_get_significant_differences_summary(self, loaded_comparator):
        t_results = loaded_comparator.perform_all_pairwise_t_tests()
        anova_results = loaded_comparator.perform_all_anova_tests()
        
        summary = loaded_comparator.get_significant_differences_summary(t_results, anova_results)
        
        assert "significant_t_tests" in summary
        assert "significant_anova" in summary
        assert "total_t_tests" in summary
        assert "total_anova_tests" in summary
        assert "significance_threshold" in summary
    
    def test_generate_report(self, loaded_comparator):
        report = loaded_comparator.generate_report()
        
        assert isinstance(report, AnalysisReport)
        assert len(report.groups_analyzed) == 3
        assert len(report.columns_analyzed) == 4
        assert isinstance(report.summary_statistics, dict)
    
    def test_export_report_to_text(self, loaded_comparator):
        report = loaded_comparator.generate_report()
        text_report = loaded_comparator.export_report_to_text(report)
        
        assert isinstance(text_report, str)
        assert "实验结果对比分析报告" in text_report
        assert "group_a" in text_report
        assert "group_b" in text_report
        assert "group_c" in text_report
        assert "t-test" in text_report
        assert "ANOVA" in text_report
    
    def test_export_report_to_text_with_output(self, loaded_comparator):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            report = loaded_comparator.generate_report()
            text_report = loaded_comparator.export_report_to_text(report, output_path)
            
            assert os.path.exists(output_path)
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert content == text_report
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_export_summary_to_csv(self, loaded_comparator):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            loaded_comparator.export_summary_to_csv(output_path)
            
            assert os.path.exists(output_path)
            df = pd.read_csv(output_path)
            
            assert len(df) > 0
            assert 'test_type' in df.columns
            assert 'column' in df.columns
            assert 'p_value' in df.columns
            assert 'is_significant' in df.columns
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_paired_t_test(self, test_data_dir):
        comp = ExperimentComparator(paired_t_test=True)
        comp.load_multiple_csv(str(test_data_dir))
        
        result = comp.perform_t_test("group_a", "group_b", "accuracy")
        
        assert isinstance(result, TTestResult)
        assert result.degrees_of_freedom == 9  # n-1 for paired test
    
    def test_t_test_significance_logic(self):
        comp = ExperimentComparator(significance_level=0.05)
        
        group1_data = pd.DataFrame({
            'value': np.random.normal(0, 1, 100)
        })
        group2_data = pd.DataFrame({
            'value': np.random.normal(0, 1, 100)
        })
        
        comp.groups['g1'] = GroupData(name='g1', data=group1_data, file_path='g1.csv')
        comp.groups['g2'] = GroupData(name='g2', data=group2_data, file_path='g2.csv')
        
        result = comp.perform_t_test('g1', 'g2', 'value')
        
        if result.p_value < 0.05:
            assert result.is_significant == True
        else:
            assert result.is_significant == False
    
    def test_anova_significance_logic(self):
        comp = ExperimentComparator(significance_level=0.05)
        
        group1_data = pd.DataFrame({'value': np.random.normal(0, 1, 100)})
        group2_data = pd.DataFrame({'value': np.random.normal(0, 1, 100)})
        group3_data = pd.DataFrame({'value': np.random.normal(0, 1, 100)})
        
        comp.groups['g1'] = GroupData(name='g1', data=group1_data, file_path='g1.csv')
        comp.groups['g2'] = GroupData(name='g2', data=group2_data, file_path='g2.csv')
        comp.groups['g3'] = GroupData(name='g3', data=group3_data, file_path='g3.csv')
        
        result = comp.perform_anova('value')
        
        if result.p_value < 0.05:
            assert result.is_significant == True
        else:
            assert result.is_significant == False
    
    def test_anova_post_hoc_tests(self):
        comp = ExperimentComparator(significance_level=0.05)
        
        np.random.seed(42)
        group1_data = pd.DataFrame({'value': np.random.normal(0, 1, 100)})
        group2_data = pd.DataFrame({'value': np.random.normal(10, 1, 100)})
        group3_data = pd.DataFrame({'value': np.random.normal(0, 1, 100)})
        
        comp.groups['g1'] = GroupData(name='g1', data=group1_data, file_path='g1.csv')
        comp.groups['g2'] = GroupData(name='g2', data=group2_data, file_path='g2.csv')
        comp.groups['g3'] = GroupData(name='g3', data=group3_data, file_path='g3.csv')
        
        result = comp.perform_anova('value')
        
        assert result.is_significant == True
        assert len(result.post_hoc_tests) > 0
        for test in result.post_hoc_tests:
            assert len(test) == 4
            assert isinstance(test[0], str)
            assert isinstance(test[1], str)
            assert isinstance(test[2], float)
            assert isinstance(test[3], bool)
