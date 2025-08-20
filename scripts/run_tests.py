"""
FlashMM Test Runner

Executes the complete test suite and generates comprehensive reports
for unit tests, integration tests, and performance validation.
"""

import subprocess
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import os

def run_command(command: List[str], description: str) -> Dict[str, Any]:
    """Run a command and capture results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        if success:
            print(f"✅ {description} PASSED ({duration:.1f}s)")
        else:
            print(f"❌ {description} FAILED ({duration:.1f}s)")
            print(f"Error output:\n{result.stderr}")
        
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        
        return {
            'description': description,
            'command': ' '.join(command),
            'success': success,
            'duration_seconds': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} TIMED OUT after 5 minutes")
        return {
            'description': description,
            'command': ' '.join(command),
            'success': False,
            'duration_seconds': 300,
            'stdout': '',
            'stderr': 'Test timed out after 5 minutes',
            'return_code': -1
        }
    except Exception as e:
        print(f"💥 {description} CRASHED: {e}")
        return {
            'description': description,
            'command': ' '.join(command),
            'success': False,
            'duration_seconds': 0,
            'stdout': '',
            'stderr': str(e),
            'return_code': -2
        }

def main():
    """Main test runner function."""
    print("""
    ███████╗██╗      █████╗ ███████╗██╗  ██╗███╗   ███╗███╗   ███╗
    ██╔════╝██║     ██╔══██╗██╔════╝██║  ██║████╗ ████║████╗ ████║
    █████╗  ██║     ███████║███████╗███████║██╔████╔██║██╔████╔██║
    ██╔══╝  ██║     ██╔══██║╚════██║██╔══██║██║╚██╔╝██║██║╚██╔╝██║
    ██║     ███████╗██║  ██║███████║██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║
    ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝
    
    COMPREHENSIVE TEST SUITE RUNNER
    """)
    
    start_time = time.time()
    test_results = []
    
    # Change to project root directory
    project_root = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(project_root)
    
    print(f"Project root: {os.getcwd()}")
    print(f"Test execution started at: {datetime.now().isoformat()}")
    
    # Test suite configuration
    test_suite = [
        {
            'command': ['python', '-m', 'pytest', 'tests/unit/', '-v', '--tb=short', '--durations=10'],
            'description': 'Unit Tests'
        },
        {
            'command': ['python', '-m', 'pytest', 'tests/integration/', '-v', '--tb=short', '--durations=10'],
            'description': 'Integration Tests'
        },
        {
            'command': ['python', '-m', 'pytest', 'tests/', '-m', 'performance', '-v', '--tb=short'],
            'description': 'Performance Tests'
        },
        {
            'command': ['python', 'scripts/validate_performance.py'],
            'description': 'Performance Validation'
        }
    ]
    
    # Execute test suite
    for test_config in test_suite:
        result = run_command(test_config['command'], test_config['description'])
        test_results.append(result)
    
    # Generate summary report
    total_duration = time.time() - start_time
    passed_tests = sum(1 for r in test_results if r['success'])
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    print(f"\n{'='*80}")
    print("TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    for result in test_results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = result['duration_seconds']
        print(f"{status} | {result['description']:<25} | {duration:6.1f}s")
    
    print(f"{'='*80}")
    print(f"TOTAL RESULTS: {passed_tests}/{total_tests} test suites passed ({success_rate:.1%})")
    print(f"TOTAL DURATION: {total_duration:.1f} seconds")
    print(f"EXECUTION TIME: {datetime.now().isoformat()}")
    
    # Overall assessment
    if success_rate >= 0.75:  # 75% pass rate required
        print("🎉 OVERALL ASSESSMENT: SYSTEM READY FOR DEPLOYMENT")
        overall_success = True
    else:
        print("⚠️  OVERALL ASSESSMENT: SYSTEM NEEDS ATTENTION BEFORE DEPLOYMENT")
        overall_success = False
    
    print(f"{'='*80}")
    
    # Generate detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'execution_duration_seconds': total_duration,
        'overall_success': overall_success,
        'summary': {
            'total_test_suites': total_tests,
            'passed_test_suites': passed_tests,
            'failed_test_suites': total_tests - passed_tests,
            'success_rate': success_rate
        },
        'test_results': test_results,
        'performance_requirements': {
            'cycle_time_target_ms': 200,
            'inventory_control_target_pct': 2.0,
            'spread_improvement_target_pct': 40.0,
            'system_uptime_target_pct': 95.0
        },
        'next_steps': generate_next_steps(test_results, overall_success)
    }
    
    # Save detailed report
    report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Detailed test report saved to: {report_filename}")
    
    # Exit with appropriate code
    return 0 if overall_success else 1

def generate_next_steps(test_results: List[Dict[str, Any]], overall_success: bool) -> List[str]:
    """Generate next steps based on test results."""
    next_steps = []
    
    if overall_success:
        next_steps.extend([
            "✅ All critical test suites passed - system is ready for deployment",
            "🚀 Consider running performance validation in production-like environment",
            "📊 Set up continuous monitoring and alerting for production deployment",
            "📋 Review performance metrics and establish baseline benchmarks",
            "🔄 Schedule regular performance validation runs"
        ])
    else:
        failed_tests = [r for r in test_results if not r['success']]
        
        for failed_test in failed_tests:
            if 'Unit Tests' in failed_test['description']:
                next_steps.append("🔧 Fix unit test failures - check component implementations")
            elif 'Integration Tests' in failed_test['description']:
                next_steps.append("🔗 Fix integration issues - check component interactions")
            elif 'Performance Tests' in failed_test['description']:
                next_steps.append("⚡ Optimize performance - review cycle times and resource usage")
            elif 'Performance Validation' in failed_test['description']:
                next_steps.append("🎯 Address performance validation failures - check system requirements")
        
        next_steps.extend([
            "⚠️  Do not deploy to production until all critical tests pass",
            "🔍 Review test logs and error messages for specific issues",
            "🛠️  Consider adjusting system parameters or optimization settings",
            "🔄 Re-run tests after fixes to confirm resolution"
        ])
    
    return next_steps

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Test runner crashed: {e}")
        sys.exit(1)