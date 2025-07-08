import json
import os
import subprocess
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Local imports
from utils import get_llm_response
from envs import DEFAULT_PATH, GENERATE_DATA_PATH, LOG_PATH
from temp_testbed import TempTestbed, get_all_filenames
from grading_simple import get_eval_report_synthetic
from gpt_1_utils import (
    # Data classes
    BugFile, ParsedBugResponse,
    # Parsing functions
    parse_bug_response, generate_patches_for_bug_data,
    # Command execution
    run_command_with_timeout, make_test_command_for_gpt,
    # File I/O
    save_checkpoint_state, load_checkpoint_state,
    save_all_tasks_with_state, load_all_tasks_with_state,
    load_processed_ids_from_output, load_perfect_tasks_from_output,
    append_results_to_jsonl,
    # Prompt reconstruction
    reconstruct_three_shot_prompt,
    # Constants
    NON_TEST_EXTS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
date = time.strftime("%Y-%m-%d")
EXP_PATH = f"{LOG_PATH}/test_log/py-gpt-bug-patch-commit_{date}_combined"
EXPR_PATH = os.getenv("EXPR_PATH", "/opt/tiger/expr")
ENV_DIR = f"{EXPR_PATH}/conda_env/"

# ================== Core Evaluation Functions ==================

def test_init(tasks, max_workers, timeout):
    """测试初始状态（修复后）GPT生成的测试是否能通过"""
    log_path = f"{EXP_PATH}/init_eval_log"
    os.makedirs(log_path, exist_ok=True)
    
    start_time = time.time()
    results = eval_parallel_init_tasks(tasks, log_path, max_workers=max_workers, timeout=timeout)
    total_time = time.time() - start_time
    
    logger.info("\nInit State Evaluation Summary:")
    logger.info(f"Total time: {total_time:.2f}s")
    if results:
        logger.info(f"Average task time: {total_time/len(results):.2f}s")
    
    # 直接保存所有结果，不进行分类
    init_results_path = f"{EXP_PATH}/init_all_results.json"
    with open(init_results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Init evaluation results saved to {init_results_path}")
    return results

def eval_parallel_init_tasks(tasks: List[Dict], log_path, max_workers: int = 4, timeout=100) -> List[Dict]:
    """并行运行多个初始状态评估任务"""
    results = {}
    complete_tasks = 0
    exec_func = eval_init_instance
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(exec_func, task, log_path, timeout): task
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            result = future.result()
            instance_id = list(result.keys())[0]
            result_value = result[instance_id]
            
            # 合并测试状态（如果实例已存在）
            if instance_id in results:
                if 'tests_status' in result_value and 'tests_status' in results[instance_id]:
                    results[instance_id]['tests_status']['PASSED'].extend(
                        result_value['tests_status']['PASSED']
                    )
                    results[instance_id]['tests_status']['FAILED'].extend(
                        result_value['tests_status']['FAILED']
                    )
            else:
                results.update(result)
                
            complete_tasks += 1
            status = "SUCCESS" if result_value["success"] else "FAILED"
            logger.info(
                f"Progress: {complete_tasks}/{len(tasks)} - "
                f"Task {instance_id} {status} - "
                f"Duration: {result_value['duration']:.2f}s"
            )
    
    return results

def eval_init_instance(instance: dict, log_path: str, timeout=100) -> dict:
    """
    评估初始状态的实例（只应用原始patch和测试patch，不应用bug patch）
    用于验证GPT生成的测试在修复状态下是否能通过
    """
    instance_id = instance['instance_id']
    repo = instance['repo']
    base_commit = instance['base_commit']
    repo_commit = repo.replace("/", "__") + "__" + base_commit[:6]
    
    # 获取patches
    original_patch = instance.get('patch', '')  # 原始修复patch
    test_patch = instance.get('test_patch', '')  # GPT生成的测试patch
    
    # 创建临时patch文件路径
    init_test_patch_path = f"{GENERATE_DATA_PATH}/init-eval/{instance_id}/test_patch.diff"
    
    # 创建目录
    os.makedirs(os.path.dirname(init_test_patch_path), exist_ok=True)
    
    # 写入patch文件
    with open(init_test_patch_path, 'w') as f:
        f.write(test_patch)
    
    # 获取需要复制的文件
    original_patch_files = get_all_filenames(original_patch)
    test_files = get_all_filenames(test_patch)
    
    # 合并所有需要的文件
    files_to_copy = list(set(
        original_patch_files["modified"] + 
        original_patch_files["added"] +
        test_files["modified"] + 
        test_files["added"]
    ))
    
    source_testbed = os.path.join(DEFAULT_PATH, repo_commit)
    conda_path = os.path.join(ENV_DIR, repo_commit)

    eval_sh = "./eval.sh"
    
    # 检查脚本是否存在
    if not os.path.exists(eval_sh):
        raise FileNotFoundError(f"Evaluation script not found: {eval_sh}")
    
    try:
        with TempTestbed(source_testbed=source_testbed, copy_files=files_to_copy) as temp_testbed:
            temp_dir = temp_testbed.temp_dir
            temp_pytest = temp_testbed.temp_pytest
            
            # 构建测试命令
            test_command = make_test_command_for_gpt(instance, conda_path, temp_pytest, test_patch, NON_TEST_EXTS)
            
            # 确保使用绝对路径
            log_path = os.path.abspath(log_path)
            os.makedirs(log_path, exist_ok=True)
            instance_log = os.path.join(log_path, f"{instance_id}.log")

            # 运行测试
            cmd = f'bash {eval_sh} {temp_dir} "{test_command}" {init_test_patch_path}'
            cmd += f' > {instance_log} 2>&1'
            
            start_time = time.time()
            try:
                # 运行命令并获取结果
                success, timed_out = run_command_with_timeout(
                    instance_id, cmd, timeout=timeout
                )
                
                report = get_eval_report_synthetic(instance, instance_log, True)
                report[instance_id]["timed_out"] = timed_out
                report[instance_id]["success"] = success
                
            except Exception as e:
                report = {
                    instance_id: {
                        "success": False,
                        "error": str(e),
                        "timed_out": False,
                        "duration": 0
                    }
                }
                    
            end_time = time.time()
            duration = end_time - start_time
            report[instance_id]["duration"] = duration
            
    except Exception as e:
        logger.error(f"Error evaluating init instance {instance_id}: {str(e)}")
        report = {
            instance_id: {
                "success": False,
                "error": str(e),
                "timed_out": False,
                "duration": 0
            }
        }
    
    return report

def test_gpt_bug(tasks, max_workers, timeout):
    """测试GPT生成的bug"""
    log_path = f"{EXP_PATH}/gpt_bug_eval_log"
    os.makedirs(log_path, exist_ok=True)
    
    start_time = time.time()
    results = eval_parallel_gpt_bug_tasks(tasks, log_path, max_workers=max_workers, timeout=timeout)
    total_time = time.time() - start_time
    
    logger.info("\nGPT Bug Evaluation Summary:")
    logger.info(f"Total time: {total_time:.2f}s")
    if results:
        logger.info(f"Average task time: {total_time/len(results):.2f}s")
    
    # 直接保存所有结果，不进行分类
    gpt_bug_results_path = f"{EXP_PATH}/gpt_bug_all_results.json"
    with open(gpt_bug_results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"GPT bug evaluation results saved to {gpt_bug_results_path}")
    return results

def eval_parallel_gpt_bug_tasks(tasks: List[Dict], log_path, max_workers: int = 4, timeout=100) -> List[Dict]:
    """并行运行多个GPT bug评估任务"""
    results = {}
    complete_tasks = 0
    exec_func = eval_gpt_bug_instance
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(exec_func, task, log_path, timeout): task
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            result = future.result()
            instance_id = list(result.keys())[0]
            result_value = result[instance_id]
            
            # 合并测试状态（如果实例已存在）
            if instance_id in results:
                if 'tests_status' in result_value and 'tests_status' in results[instance_id]:
                    results[instance_id]['tests_status']['PASSED'].extend(
                        result_value['tests_status']['PASSED']
                    )
                    results[instance_id]['tests_status']['FAILED'].extend(
                        result_value['tests_status']['FAILED']
                    )
            else:
                results.update(result)
                
            complete_tasks += 1
            status = "SUCCESS" if result_value["success"] else "FAILED"
            logger.info(
                f"Progress: {complete_tasks}/{len(tasks)} - "
                f"Task {instance_id} {status} - "
                f"Duration: {result_value['duration']:.2f}s"
            )
    
    return results

def eval_gpt_bug_instance(instance: dict, log_path: str, timeout=100) -> dict:
    """评估GPT生成的bug实例"""
    instance_id = instance['instance_id']
    repo = instance['repo']
    base_commit = instance['base_commit']
    repo_commit = repo.replace("/", "__") + "__" + base_commit[:6]
    
    # 获取GPT生成的patches
    gpt_patch = instance.get('gpt_patch', '')  # GPT生成的bug patch
    
    if not gpt_patch:
        return {
            instance_id: {
                "success": False,
                "error": "Missing gpt_patch",
                "timed_out": False,
                "duration": 0
            }
        }
    
    # 创建临时patch文件路径
    gpt_bug_patch_path = f"{GENERATE_DATA_PATH}/gpt-bug-eval/{instance_id}/gpt_bug_patch.diff"
    init_test_patch_path = f"{GENERATE_DATA_PATH}/gpt-init-eval/{instance_id}/test_patch.diff"
    
    # 创建目录
    os.makedirs(os.path.dirname(gpt_bug_patch_path), exist_ok=True)
    
    # 写入patch文件
    with open(gpt_bug_patch_path, 'w') as f:
        f.write(gpt_patch)
    
    # 获取需要复制的文件
    gpt_patch_files = get_all_filenames(gpt_patch)
    
    # 合并所有需要的文件
    files_to_copy = list(set(
        gpt_patch_files["modified"] + 
        gpt_patch_files["added"]
    ))
    
    source_testbed = os.path.join(DEFAULT_PATH, repo_commit)
    conda_path = os.path.join(ENV_DIR, repo_commit)
    
    # 使用现有的eval.sh脚本
    eval_sh = "./eval.sh"
    
    # 检查脚本是否存在
    if not os.path.exists(eval_sh):
        raise FileNotFoundError(f"Evaluation script not found: {eval_sh}")
    
    try:
        with TempTestbed(source_testbed=source_testbed, copy_files=files_to_copy) as temp_testbed:
            temp_dir = temp_testbed.temp_dir
            temp_pytest = temp_testbed.temp_pytest
            
            # 构建测试命令
            test_command = make_test_command_for_gpt(instance, conda_path, temp_pytest, instance['test_patch'], NON_TEST_EXTS)
            
            # 确保使用绝对路径
            log_path = os.path.abspath(log_path)
            os.makedirs(log_path, exist_ok=True)
            instance_log = os.path.join(log_path, f"{instance_id}.log")
            
            # 应用GPT bug patch (逆向，引入bug)
            patch_cmd_2 = f'cd {temp_dir} && git apply --reverse --whitespace=nowarn {gpt_bug_patch_path}'
            result_2 = subprocess.run(patch_cmd_2, shell=True, capture_output=True, text=True)
            if result_2.returncode != 0:
                return {
                    instance_id: {
                        "success": False,
                        "error": f"Failed to apply GPT bug patch: {result_2.stderr}",
                        "timed_out": False,
                        "duration": 0
                    }
                }

            # 运行测试
            cmd = f'bash {eval_sh} {temp_dir} "{test_command}" {init_test_patch_path}'
            cmd += f' > {instance_log} 2>&1'
            
            start_time = time.time()
            try:
                # 运行命令并获取结果
                success, timed_out = run_command_with_timeout(
                    instance_id, cmd, timeout=timeout
                )
                
                report = get_eval_report_synthetic(instance, instance_log, True)
                report[instance_id]["timed_out"] = timed_out
                report[instance_id]["success"] = success
                
            except Exception as e:
                report = {
                    instance_id: {
                        "success": False,
                        "error": str(e),
                        "timed_out": False,
                        "duration": 0
                    }
                }
                    
            end_time = time.time()
            duration = end_time - start_time
            report[instance_id]["duration"] = duration
            
    except Exception as e:
        logger.error(f"Error evaluating GPT bug instance {instance_id}: {str(e)}")
        report = {
            instance_id: {
                "success": False,
                "error": str(e),
                "timed_out": False,
                "duration": 0
            }
        }
    
    return report

def analyze_combined_results(init_results, gpt_bug_results):
    """分析合并的结果，筛选出符合条件的测试"""
    logger.info("Analyzing combined results...")
    
    # 找到两个结果集的交集
    common_instances = set(init_results.keys()) & set(gpt_bug_results.keys())
    logger.info(f"Common instances between init and gpt_bug: {len(common_instances)}")
    
    # 符合条件的测试：init通过 + gpt_bug失败
    good_tests = {}
    # init失败的测试
    init_failed = {}
    # gpt_bug通过的测试（没有检测到bug）
    bug_not_detected = {}
    # 其他情况
    other_cases = {}
    
    for instance_id in common_instances:
        init_result = init_results[instance_id]
        bug_result = gpt_bug_results[instance_id]
        
        # 获取init状态和bug状态的测试状态
        init_passed_tests = set(init_result.get('tests_status', {}).get('PASSED', []))
        init_failed_tests = set(init_result.get('tests_status', {}).get('FAILED', []))
        
        bug_passed_tests = set(bug_result.get('tests_status', {}).get('PASSED', []))
        bug_failed_tests = set(bug_result.get('tests_status', {}).get('FAILED', []))
        
        # 找到在init状态通过但在bug状态失败的测试
        perfect_test_cases = init_passed_tests & bug_failed_tests
        
        if perfect_test_cases:
            # 存在至少一个完美的测试：init通过，bug被检测到
            good_tests[instance_id] = {
                'init_result': init_result,
                'bug_result': bug_result,
                'perfect_tests': list(perfect_test_cases),  # 只保留符合条件的测试
                'category': 'perfect'
            }
        elif not init_result["success"] or not init_passed_tests:
            # init运行失败或没有通过的测试
            init_failed[instance_id] = {
                'init_result': init_result,
                'bug_result': bug_result,
                'category': 'init_failed'
            }
        elif init_passed_tests and (not bug_result["success"] or init_passed_tests.issubset(bug_passed_tests)):
            # init有通过的测试，但bug状态下这些测试仍然通过（没检测到bug）
            bug_not_detected[instance_id] = {
                'init_result': init_result,
                'bug_result': bug_result,
                'category': 'bug_not_detected'
            }
        else:
            # 其他情况
            other_cases[instance_id] = {
                'init_result': init_result,
                'bug_result': bug_result,
                'category': 'other'
            }
    
    # 打印统计信息
    logger.info(f"Perfect tests (at least one test: init pass + bug detected): {len(good_tests)}")
    logger.info(f"Init failed tests: {len(init_failed)}")
    logger.info(f"Bug not detected tests: {len(bug_not_detected)}")
    logger.info(f"Other cases: {len(other_cases)}")
    
    # 保存分类结果
    perfect_tests_path = f"{EXP_PATH}/perfect_tests.json"
    with open(perfect_tests_path, "w") as f:
        json.dump(good_tests, f, indent=4)
    
    init_failed_path = f"{EXP_PATH}/init_failed_tests.json"
    with open(init_failed_path, "w") as f:
        json.dump(init_failed, f, indent=4)
    
    bug_not_detected_path = f"{EXP_PATH}/bug_not_detected_tests.json"
    with open(bug_not_detected_path, "w") as f:
        json.dump(bug_not_detected, f, indent=4)
    
    other_cases_path = f"{EXP_PATH}/other_cases_tests.json"
    with open(other_cases_path, "w") as f:
        json.dump(other_cases, f, indent=4)
    
    # 计算总的完美测试数量
    total_perfect_tests = sum(len(result.get('perfect_tests', [])) for result in good_tests.values())
    
    # 保存最终综合结果
    final_results = {
        'summary': {
            'total_common_instances': len(common_instances),
            'perfect_instances_count': len(good_tests),
            'total_perfect_tests_count': total_perfect_tests,
            'init_failed_count': len(init_failed),
            'bug_not_detected_count': len(bug_not_detected),
            'other_cases_count': len(other_cases),
            'success_rate': len(good_tests) / len(common_instances) if common_instances else 0
        },
        'categorized_results': {
            'perfect_tests': good_tests,
            'init_failed': init_failed,
            'bug_not_detected': bug_not_detected,
            'other_cases': other_cases
        },
        'analysis_metadata': {
            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'criteria': {
                'perfect_test': 'At least one test: init_pass=True AND bug_detected=True',
                'init_failed': 'init_success=False OR no passed tests in init',
                'bug_not_detected': 'init has passed tests BUT all init_passed tests still pass in bug state',
                'other_cases': 'All other combinations'
            }
        }
    }
    
    final_results_path = f"{EXP_PATH}/final_analysis_results.json"
    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    
    logger.info("Analysis results saved.")
    
    return {
        'perfect_tests': good_tests,
        'init_failed': init_failed,
        'bug_not_detected': bug_not_detected,
        'other_cases': other_cases,
        'summary': final_results['summary']
    }

# ================== Main Pipeline Functions ==================

def run_extraction_phase(tasks: List[dict]) -> Tuple[List[dict], set]:
    """
    Takes a list of tasks, parses the LLM response, and generates patches.
    
    Returns:
        A tuple containing:
        - A list of successfully processed tasks with patches.
        - A set of instance_ids for tasks that failed to parse.
    """
    logger.info("=== Phase 1: Extracting bugs and generating patches ===")
    extracted_tasks = []
    failed_ids = set()

    for task in tasks:
        instance_id = task.get('instance_id', 'unknown')
        try:
            # The response is inside 'meta_response'
            meta_response = json.loads(task.get('meta_response'))
            response = meta_response['choices'][0]['message']['content']
            
            if not isinstance(response, str) or not response:
                logger.warning(f"Skipping {instance_id}: Response is not a valid string.")
                failed_ids.add(instance_id)
                continue
                
            result = parse_bug_response(response)
            if result:
                processed_data = generate_patches_for_bug_data(task, result, DEFAULT_PATH)
                # Remove large fields that are no longer needed
                processed_data.pop('api_resposne', None) 
                processed_data.pop('meta_response', None)
                extracted_tasks.append(processed_data)
            else:
                logger.warning(f"Failed to parse bug response for {instance_id}")
                failed_ids.add(instance_id)
                
        except Exception as e:
            logger.error(f"Error processing task {instance_id} during extraction: {str(e)}")
            failed_ids.add(instance_id)
            continue
            
    logger.info(f"Extraction phase summary: {len(extracted_tasks)} succeeded, {len(failed_ids)} failed parsing.")
    return extracted_tasks, failed_ids

def run_evaluation_phase(tasks_to_evaluate: List[dict], is_test_mode: bool):
    """
    Runs the full evaluation pipeline (init, gpt_bug) and analysis.
    
    Returns:
        The dictionary of analysis results.
    """
    logger.info(f"=== Phase 2: Evaluating patches for {len(tasks_to_evaluate)} tasks ===")
    
    if is_test_mode:
        EXP_PATH_TEST = '/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data/test_log/py-gpt-bug-patch-commit_2025-07-03-07_combined'
        final_results_path = f"{EXP_PATH_TEST}/final_analysis_results.json"
        logger.info(f"TEST MODE: Loading existing results from {final_results_path}")
        with open(final_results_path, "r") as f:
            analysis_results = json.load(f)
        return analysis_results.get('categorized_results', {
            'perfect_tests': {}, 'init_failed': {}, 'bug_not_detected': {}, 'other_cases': {}
        })

    # Production mode: run the evaluations
    logger.info("Starting init state evaluation...")
    init_results = test_init(tasks_to_evaluate, 50, 45)
    
    logger.info("Starting GPT bug evaluation...")
    gpt_bug_results = test_gpt_bug(tasks_to_evaluate, 50, 45)
    
    logger.info("Analyzing combined results...")
    analysis_results = analyze_combined_results(init_results, gpt_bug_results)
    
    return analysis_results

def process_single_task_with_reconstruction(task: dict, all_perfect_tasks: list, repo_path: str) -> dict:
    """
    Attempts to reconstruct the prompt for a task using the new function.
    If successful, uses the reconstructed prompt to get a new response.
    Otherwise, falls back to the original method.
    """
    try:
        # Attempt to reconstruct the prompt
        # logger.info(f"Attempting prompt reconstruction for task {task.get('instance_id', 'unknown')}")
        
        # Use the new reconstruction function
        reconstructed_prompt = reconstruct_three_shot_prompt(
            task_item=task,
            repo_path=repo_path,
            sample_data=all_perfect_tasks
        )
        
        if reconstructed_prompt and reconstructed_prompt.strip():
            # logger.info(f"Successfully reconstructed prompt for task {task.get('instance_id', 'unknown')}")
            
            # Get a new response using the reconstructed prompt
            new_response = get_llm_response(reconstructed_prompt)
            
            # Update the task with the new prompt and response
            task_copy = task.copy()
            task_copy['messages'] = [{"role": "user", "content": reconstructed_prompt}]
            task_copy['response'] = new_response
            
            # logger.info(f"Generated new response for task {task.get('instance_id', 'unknown')}")
            return task_copy
        else:
            logger.warning(f"Prompt reconstruction returned empty for task {task.get('instance_id', 'unknown')}, using original")
            return task
            
    except Exception as e:
        logger.error(f"Failed to reconstruct prompt for task {task.get('instance_id', 'unknown')}: {e}")
        return task

# ================== Main Execution ==================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract bugs and evaluate GPT bug detection with retry mechanism.')
    parser.add_argument('--test', action='store_true', 
                       help='Use test mode for evaluation (load existing results)')
    parser.add_argument('--max_retries', type=int, default=5, 
                        help='Maximum number of retries for failed items (default: 5). Set to 1 to disable retries.')
    parser.add_argument('--enable_retry', action='store_true', 
                        help='Enable the retry mechanism for failed items. If not set, runs only once.')
    parser.add_argument('--restart', action='store_true',
                        help='Restart from saved checkpoint if available')

    args = parser.parse_args()
    
    start_time = time.time()

    # --- Define output file ---
    final_output_file = f"{GENERATE_DATA_PATH}/gpt_2_finish_bug_gpt4o.jsonl"

    # --- Load Initial Data ---
    # Always load initial data from the source file
    initial_tasks = []
    input_jsonl_file = f'{GENERATE_DATA_PATH}/gpt-4o-2024-11-20_yimi_three_shot_same_test.jsonl.tmp'
    if not os.path.exists(input_jsonl_file):
        print(f"Warning: Input file not found: {input_jsonl_file}. Skipping.")
        input_jsonl_file = input_jsonl_file.replace(".tmp", "")
    try:
        with open(input_jsonl_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                initial_tasks.append(json.loads(line))
        logger.info(f"Successfully loaded {len(initial_tasks)} initial tasks from {input_jsonl_file}")
    except FileNotFoundError:
        logger.error(f"FATAL: Initial data file not found at {input_jsonl_file}. Exiting.")
        sys.exit(1)

    # --- Initialize state variables ---
    if args.restart:
        logger.info("Restart mode enabled. Loading from existing output file...")
        
        # Load already processed task IDs and perfect tasks from output file
        processed_ids = load_processed_ids_from_output(final_output_file)
        all_perfect_tasks = load_perfect_tasks_from_output(final_output_file)
        
        # 重启模式下，不需要重建all_perfect_results，因为我们使用增量保存
        all_perfect_results = {}  # 只用于跟踪新的结果
        
        # Build tasks_to_process: exclude already processed items
        tasks_to_process = [
            task for task in initial_tasks 
            if task['instance_id'] not in processed_ids
        ]
        tasks_to_process = tasks_to_process[:500]
        logger.info(f"Restarting with fresh attempt counting (max_retries: {args.max_retries})")
        logger.info(f"Already processed: {len(processed_ids)} tasks")
        logger.info(f"Remaining to process: {len(tasks_to_process)} tasks")
        
    else:
        # Normal initialization
        tasks_to_process = initial_tasks
        all_perfect_results = {}
        all_perfect_tasks = []
        processed_ids = set()

    # 每次都重新开始计算尝试次数
    max_attempts = args.max_retries if args.enable_retry else 1
    for attempt in range(1, max_attempts + 1):
        if not tasks_to_process:
            logger.info("No more tasks to process. Exiting retry loop.")
            break
        
        logger.info(f"--- Starting Attempt {attempt}/{max_attempts} ---")
        logger.info(f"Processing {len(tasks_to_process)} tasks.")

        # If this is a retry (attempt > 1) OR restart mode, regenerate the LLM response.
        if attempt > 1 or args.restart:
            if args.restart and attempt == 1:
                logger.info("Restart mode: Regenerating LLM responses for remaining tasks...")
            else:
                logger.info("Regenerating LLM responses for failed tasks...")
            
            # Use ThreadPoolExecutor for concurrent requests with progress bar
            max_workers = min(20, len(tasks_to_process))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks with reconstruction
                future_to_task = {
                    executor.submit(process_single_task_with_reconstruction, task, initial_tasks, DEFAULT_PATH): task
                    for task in tasks_to_process
                }
                
                # Process completed tasks with progress bar
                completed_tasks = []
                desc = "Regenerating LLM responses (restart)" if args.restart and attempt == 1 else "Regenerating LLM responses (retry)"
                with tqdm(total=len(tasks_to_process), desc=desc, unit="task") as pbar:
                    for future in as_completed(future_to_task):
                        try:
                            completed_task = future.result()
                            completed_tasks.append(completed_task)
                        except Exception as e:
                            original_task = future_to_task[future]
                            logger.error(f"Failed to process task {original_task.get('instance_id', 'unknown')}: {e}")
                            # Keep the original task if processing failed
                            completed_tasks.append(original_task)
                        finally:
                            pbar.update(1)
                
                # Update tasks_to_process with the completed tasks
                tasks_to_process = completed_tasks
        
        # Phase 1: Extract Patches from LLM responses
        extracted_tasks, failed_to_parse_ids = run_extraction_phase(tasks_to_process)
        
        # If nothing was extracted, all tasks failed parsing. Prepare them for the next retry.
        if not extracted_tasks:
            logger.warning("No tasks were successfully extracted in this attempt.")
            # Update tasks_to_process for the next loop
            tasks_to_process = [
                task for task in initial_tasks 
                if task['instance_id'] in failed_to_parse_ids and task['instance_id'] not in processed_ids
            ]
            continue

        # Phase 2: Evaluate successfully extracted patches
        analysis_results = run_evaluation_phase(extracted_tasks, args.test)

        # --- Process and Segregate Results of the Current Iteration ---
        perfect_results_this_iter = analysis_results.get('perfect_tests', {})
        logger.info(f"Attempt {attempt}: Found {len(perfect_results_this_iter)} new instances with perfect tests.")
        
        # Add successful results to the main collection
        all_perfect_results.update(perfect_results_this_iter)
        
        # Collect the full task data for the successful items and mark as processed
        perfect_ids_this_iter = set(perfect_results_this_iter.keys())
        for task in extracted_tasks:
            if task['instance_id'] in perfect_ids_this_iter and task['instance_id'] not in processed_ids:
                all_perfect_tasks.append(task)
                processed_ids.add(task['instance_id'])

        # --- Save progress after each iteration ---
        if perfect_results_this_iter:  # 只保存本次迭代新产生的完美结果
            # 只为本次迭代的成功任务保存结果
            current_iter_perfect_tasks = [
                task for task in extracted_tasks 
                if task['instance_id'] in perfect_ids_this_iter and task['instance_id'] not in processed_ids
            ]
            
            if current_iter_perfect_tasks:
                # 追加保存本次的结果到输出文件（无缝衔接已有数据）
                append_results_to_jsonl(
                    perfect_results_this_iter,
                    current_iter_perfect_tasks,
                    final_output_file
                )
                logger.info(f"Progress saved: {len(current_iter_perfect_tasks)} new perfect tasks appended to {final_output_file}")

        # --- Prepare for the Next Iteration ---
        # Identify all tasks that were evaluated in this iteration
        evaluated_ids_this_iter = set(perfect_results_this_iter.keys()) | \
                                  set(analysis_results.get('init_failed', {}).keys()) | \
                                  set(analysis_results.get('bug_not_detected', {}).keys()) | \
                                  set(analysis_results.get('other_cases', {}).keys())

        # An evaluation failure is an evaluated task that wasn't 'perfect'
        evaluation_failed_ids = evaluated_ids_this_iter - perfect_ids_this_iter
        
        # A task fails this iteration if it failed parsing OR failed evaluation
        all_failed_ids_this_iter = evaluation_failed_ids | failed_to_parse_ids
        
        # Filter the original complete task list to get the ones that need retrying
        tasks_to_process = [
            task for task in initial_tasks 
            if task['instance_id'] in all_failed_ids_this_iter and task['instance_id'] not in processed_ids
        ]
        
        logger.info(f"End of Attempt {attempt}. Total successful instances: {len(all_perfect_tasks)}.")
        if attempt < max_attempts:
            logger.info(f"{len(tasks_to_process)} tasks remaining for next attempt.")

    # --- Finalization after all attempts ---
    logger.info("="*20 + " All Attempts Finished " + "="*20)

    # 统计最终结果（从文件读取，包括之前和本次处理的所有数据）
    final_all_perfect_tasks = load_perfect_tasks_from_output(final_output_file)
    
    if final_all_perfect_tasks:
        logger.info(f"Final analysis complete. Log files saved to {EXP_PATH}")
        logger.info(f"Total unique perfect instances found across all attempts: {len(final_all_perfect_tasks)}")
        logger.info(f"Final unittest data saved to: {final_output_file}")
        logger.info(f"Final data count: {len(final_all_perfect_tasks)}")
    else:
        logger.warning("No 'perfect tests' were found in any attempt. The output file will be empty.")
        # Create an empty file to signify completion
        open(final_output_file, 'w').close()

    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f}s")