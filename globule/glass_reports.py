"""Glass Engine Report Generation System."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .glass import TestResult, TestRunResult, TestStatus


class ReportGenerator:
    """Generator for various test report formats."""
    
    def __init__(self):
        self.console = None
        try:
            from rich.console import Console
            self.console = Console()
        except ImportError:
            pass
    
    def generate_summary_report(self, run_result: TestRunResult) -> str:
        """Generate a markdown summary report."""
        lines = []
        
        # Header
        lines.append("# Glass Engine Test Run Summary")
        lines.append("")
        lines.append(f"**Mode:** {run_result.mode.title()}")
        lines.append(f"**Run ID:** {run_result.run_id}")
        lines.append(f"**Duration:** {run_result.total_duration:.2f}s")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Results overview
        lines.append("## Results Overview")
        lines.append("")
        lines.append(f"- **Total Tests:** {run_result.total_tests}")
        lines.append(f"- **Passed:** {run_result.passed_tests}")
        lines.append(f"- **Failed:** {run_result.failed_tests}")
        lines.append(f"- **Errors:** {run_result.error_tests}")
        lines.append(f"- **Skipped:** {run_result.skipped_tests}")
        lines.append("")
        
        # Success rate
        if run_result.total_tests > 0:
            success_rate = (run_result.passed_tests / run_result.total_tests) * 100
            lines.append(f"**Success Rate:** {success_rate:.1f}%")
            lines.append("")
        
        # Test results
        lines.append("## Test Results")
        lines.append("")
        
        for result in run_result.test_results:
            status_icon = self._get_status_icon(result.status)
            lines.append(f"### {status_icon} {result.test_case_id}")
            lines.append(f"- **Status:** {result.status.value.upper()}")
            lines.append(f"- **Duration:** {result.duration:.2f}s")
            lines.append(f"- **Trace ID:** {result.trace_id}")
            
            if result.step_results:
                passed_steps = sum(1 for r in result.step_results if r.status == TestStatus.PASSED)
                total_steps = len(result.step_results)
                lines.append(f"- **Steps:** {passed_steps}/{total_steps} passed")
            
            if result.assertion_results:
                passed_assertions = sum(1 for r in result.assertion_results if r.status == TestStatus.PASSED)
                total_assertions = len(result.assertion_results)
                lines.append(f"- **Assertions:** {passed_assertions}/{total_assertions} passed")
            
            if result.error_message:
                lines.append(f"- **Error:** {result.error_message}")
            
            lines.append(f"- **Artifacts:** `{result.artifacts_path}`")
            lines.append("")
        
        # Failed tests details
        failed_tests = [r for r in run_result.test_results if r.status in [TestStatus.FAILED, TestStatus.ERROR]]
        if failed_tests:
            lines.append("## Failed Tests Details")
            lines.append("")
            
            for result in failed_tests:
                lines.append(f"### {result.test_case_id}")
                lines.append("")
                
                # Step failures
                failed_steps = [r for r in result.step_results if r.status != TestStatus.PASSED]
                if failed_steps:
                    lines.append("**Failed Steps:**")
                    for step in failed_steps:
                        lines.append(f"- Step {step.step_index + 1}: {step.error}")
                    lines.append("")
                
                # Assertion failures
                failed_assertions = [r for r in result.assertion_results if r.status != TestStatus.PASSED]
                if failed_assertions:
                    lines.append("**Failed Assertions:**")
                    for assertion in failed_assertions:
                        lines.append(f"- {assertion.message}")
                        if assertion.actual_value and assertion.expected_value:
                            lines.append(f"  - Expected: `{assertion.expected_value}`")
                            lines.append(f"  - Actual: `{assertion.actual_value}`")
                    lines.append("")
                
                lines.append(f"**Debug:** Check artifacts at `{result.artifacts_path}`")
                lines.append("")
        
        # Recommendations
        if run_result.failed_tests > 0 or run_result.error_tests > 0:
            lines.append("## Recommendations")
            lines.append("")
            
            # Analyze common failure patterns
            recommendations = self._generate_recommendations(run_result)
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*Generated by Glass Engine*")
        
        return "\n".join(lines)
    
    def generate_json_report(self, run_result: TestRunResult) -> str:
        """Generate a JSON report for programmatic consumption."""
        report_data = {
            "run_id": run_result.run_id,
            "mode": run_result.mode,
            "timestamp": datetime.now().isoformat(),
            "duration": run_result.total_duration,
            "summary": {
                "total_tests": run_result.total_tests,
                "passed": run_result.passed_tests,
                "failed": run_result.failed_tests,
                "errors": run_result.error_tests,
                "skipped": run_result.skipped_tests,
                "success_rate": (run_result.passed_tests / run_result.total_tests * 100) if run_result.total_tests > 0 else 0
            },
            "tests": []
        }
        
        for result in run_result.test_results:
            test_data = {
                "test_case_id": result.test_case_id,
                "status": result.status.value,
                "duration": result.duration,
                "trace_id": result.trace_id,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "artifacts_path": result.artifacts_path,
                "steps": {
                    "total": len(result.step_results),
                    "passed": sum(1 for r in result.step_results if r.status == TestStatus.PASSED),
                    "failed": sum(1 for r in result.step_results if r.status != TestStatus.PASSED),
                    "details": [
                        {
                            "index": r.step_index,
                            "status": r.status.value,
                            "duration": r.duration,
                            "error": r.error
                        }
                        for r in result.step_results
                    ]
                },
                "assertions": {
                    "total": len(result.assertion_results),
                    "passed": sum(1 for r in result.assertion_results if r.status == TestStatus.PASSED),
                    "failed": sum(1 for r in result.assertion_results if r.status != TestStatus.PASSED),
                    "details": [
                        {
                            "message": r.message,
                            "status": r.status.value,
                            "duration": r.duration,
                            "actual_value": r.actual_value,
                            "expected_value": r.expected_value
                        }
                        for r in result.assertion_results
                    ]
                }
            }
            
            if result.error_message:
                test_data["error_message"] = result.error_message
            
            if result.failure_details:
                test_data["failure_details"] = result.failure_details
            
            report_data["tests"].append(test_data)
        
        return json.dumps(report_data, indent=2)
    
    def generate_junit_xml(self, run_result: TestRunResult) -> str:
        """Generate a JUnit XML report for CI/CD integration."""
        lines = []
        
        # XML header
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(f'<testsuite name="Glass Engine {run_result.mode.title()} Tests"')
        lines.append(f'           tests="{run_result.total_tests}"')
        lines.append(f'           failures="{run_result.failed_tests}"')
        lines.append(f'           errors="{run_result.error_tests}"')
        lines.append(f'           skipped="{run_result.skipped_tests}"')
        lines.append(f'           time="{run_result.total_duration:.3f}"')
        lines.append(f'           timestamp="{datetime.now().isoformat()}">')
        
        # Test cases
        for result in run_result.test_results:
            lines.append(f'  <testcase name="{result.test_case_id}"')
            lines.append(f'            classname="GlassEngine.{run_result.mode.title()}"')
            lines.append(f'            time="{result.duration:.3f}">')
            
            if result.status == TestStatus.FAILED:
                lines.append(f'    <failure message="{result.error_message or "Test failed"}">')
                lines.append(f'      <![CDATA[')
                lines.append(f'Trace ID: {result.trace_id}')
                lines.append(f'Artifacts: {result.artifacts_path}')
                
                # Add step failures
                failed_steps = [r for r in result.step_results if r.status != TestStatus.PASSED]
                if failed_steps:
                    lines.append(f'Failed Steps:')
                    for step in failed_steps:
                        lines.append(f'  Step {step.step_index + 1}: {step.error}')
                
                # Add assertion failures
                failed_assertions = [r for r in result.assertion_results if r.status != TestStatus.PASSED]
                if failed_assertions:
                    lines.append(f'Failed Assertions:')
                    for assertion in failed_assertions:
                        lines.append(f'  {assertion.message}')
                        if assertion.actual_value and assertion.expected_value:
                            lines.append(f'    Expected: {assertion.expected_value}')
                            lines.append(f'    Actual: {assertion.actual_value}')
                
                lines.append(f'      ]]>')
                lines.append(f'    </failure>')
                
            elif result.status == TestStatus.ERROR:
                lines.append(f'    <error message="{result.error_message or "Test error"}">')
                lines.append(f'      <![CDATA[')
                lines.append(f'Trace ID: {result.trace_id}')
                lines.append(f'Artifacts: {result.artifacts_path}')
                if result.failure_details:
                    lines.append(f'Details: {result.failure_details}')
                lines.append(f'      ]]>')
                lines.append(f'    </error>')
                
            elif result.status == TestStatus.SKIPPED:
                lines.append(f'    <skipped message="Test skipped"/>')
            
            lines.append('  </testcase>')
        
        lines.append('</testsuite>')
        
        return '\n'.join(lines)
    
    def save_reports(self, run_result: TestRunResult, output_dir: str) -> Dict[str, str]:
        """Save all report formats to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Markdown summary
        summary_file = output_path / "summary.md"
        summary_content = self.generate_summary_report(run_result)
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        saved_files["summary"] = str(summary_file)
        
        # JSON report
        json_file = output_path / "report.json"
        json_content = self.generate_json_report(run_result)
        with open(json_file, 'w') as f:
            f.write(json_content)
        saved_files["json"] = str(json_file)
        
        # JUnit XML
        junit_file = output_path / "junit.xml"
        junit_content = self.generate_junit_xml(run_result)
        with open(junit_file, 'w') as f:
            f.write(junit_content)
        saved_files["junit"] = str(junit_file)
        
        return saved_files
    
    def _get_status_icon(self, status: TestStatus) -> str:
        """Get icon for test status."""
        icons = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.ERROR: "⚠️",
            TestStatus.TIMEOUT: "⏱️",
            TestStatus.SKIPPED: "⏭️"
        }
        return icons.get(status, "❓")
    
    def _generate_recommendations(self, run_result: TestRunResult) -> List[str]:
        """Generate recommendations based on test failures."""
        recommendations = []
        
        # Analyze failure patterns
        error_patterns = {}
        timeout_count = 0
        
        for result in run_result.test_results:
            if result.status == TestStatus.TIMEOUT:
                timeout_count += 1
            
            if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                # Analyze step failures
                for step in result.step_results:
                    if step.error:
                        error_type = step.error.split(':')[0] if ':' in step.error else step.error
                        error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
                
                # Analyze assertion failures
                for assertion in result.assertion_results:
                    if assertion.status != TestStatus.PASSED:
                        error_type = assertion.message.split(':')[0] if ':' in assertion.message else assertion.message
                        error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        # Generate recommendations based on patterns
        if timeout_count > 0:
            recommendations.append(f"Consider increasing timeout values - {timeout_count} tests timed out")
        
        if "Network" in str(error_patterns):
            recommendations.append("Check network connectivity and service availability")
        
        if "Database" in str(error_patterns):
            recommendations.append("Verify database configuration and connection settings")
        
        if "Parser" in str(error_patterns):
            recommendations.append("Check LLM service availability and configuration")
        
        if "Embedding" in str(error_patterns):
            recommendations.append("Verify embedding service configuration")
        
        # Most common error
        if error_patterns:
            most_common_error = max(error_patterns.items(), key=lambda x: x[1])
            if most_common_error[1] > 1:
                recommendations.append(f"Most common issue: {most_common_error[0]} ({most_common_error[1]} occurrences)")
        
        # General recommendations
        if run_result.failed_tests > run_result.passed_tests:
            recommendations.append("Consider running in showcase mode for more detailed debugging information")
        
        return recommendations


# Utility functions
def generate_test_report(run_result: TestRunResult, format: str = "markdown") -> str:
    """Generate a test report in the specified format."""
    generator = ReportGenerator()
    
    if format == "markdown":
        return generator.generate_summary_report(run_result)
    elif format == "json":
        return generator.generate_json_report(run_result)
    elif format == "junit":
        return generator.generate_junit_xml(run_result)
    else:
        raise ValueError(f"Unsupported report format: {format}")


def save_test_reports(run_result: TestRunResult, output_dir: str) -> Dict[str, str]:
    """Save test reports in all formats."""
    generator = ReportGenerator()
    return generator.save_reports(run_result, output_dir)