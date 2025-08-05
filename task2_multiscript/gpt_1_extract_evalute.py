import json
import os
import subprocess
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

# Local imports
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
    process_single_task_with_reconstruction,
    # Evaluation utilities
    create_temp_patch_file, get_repo_commit_name, merge_files_to_copy,
    create_error_report, run_parallel_tasks, save_evaluation_results,
    retry_unittest_generation_for_task, retry_tasks_in_parallel, retry_buggy_code_in_parallel,
    # Constants
    NON_TEST_EXTS, CONC, TEST_N
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
    results = run_parallel_tasks(tasks, log_path, eval_init_instance, max_workers=max_workers, timeout=timeout)
    total_time = time.time() - start_time
    
    logger.info("\nInit State Evaluation Summary:")
    logger.info(f"Total time: {total_time:.2f}s")
    if results:
        logger.info(f"Average task time: {total_time/len(results):.2f}s")
    
    # 保存所有结果
    init_results_path = f"{EXP_PATH}/init_all_results.json"
    save_evaluation_results(results, init_results_path, "Init")
    
    return results

def eval_init_instance(instance: dict, log_path: str, timeout=100) -> dict:
    """
    评估初始状态的实例（只应用原始patch和测试patch，不应用bug patch）
    用于验证GPT生成的测试在修复状态下是否能通过
    """
    instance_id = instance['instance_id']
    new_instance_id = instance['new_instance_id']
    repo = instance['repo']
    base_commit = instance['base_commit']
    repo_commit = get_repo_commit_name(repo, base_commit)
    
    # 获取patches
    original_patch = instance.get('patch', '')  # 原始修复patch
    test_patch = instance.get('test_patch', '')  # GPT生成的测试patch
    
    # 创建临时patch文件
    init_test_patch_path = create_temp_patch_file(test_patch, GENERATE_DATA_PATH, new_instance_id, "init")
    
    # 获取需要复制的文件
    original_patch_files = get_all_filenames(original_patch)
    test_files = get_all_filenames(test_patch)
    files_to_copy = merge_files_to_copy([original_patch_files, test_files])
    
    source_testbed = os.path.join(DEFAULT_PATH, repo_commit)
    conda_path = os.path.join(ENV_DIR, repo_commit)
    eval_sh = "./eval.sh"
    
    # 检查脚本是否存在
    if not os.path.exists(eval_sh):
        return create_error_report(new_instance_id, f"Evaluation script not found: {eval_sh}")
    
    try:
        with TempTestbed(source_testbed=source_testbed, copy_files=files_to_copy) as temp_testbed:
            temp_dir = temp_testbed.temp_dir
            temp_pytest = temp_testbed.temp_pytest
            
            # 构建测试命令
            test_command = make_test_command_for_gpt(instance, conda_path, temp_pytest, test_patch, NON_TEST_EXTS)
            
            # 确保使用绝对路径
            log_path = os.path.abspath(log_path)
            os.makedirs(log_path, exist_ok=True)
            instance_log = os.path.join(log_path, f"{new_instance_id}.log")

            # 运行测试
            cmd = f'bash {eval_sh} {temp_dir} "{test_command}" {init_test_patch_path}'
            cmd += f' > {instance_log} 2>&1'
            
            start_time = time.time()
            try:
                # 运行命令并获取结果
                success, timed_out = run_command_with_timeout(
                    new_instance_id, cmd, timeout=timeout
                )
                
                report = get_eval_report_synthetic(instance, instance_log, True, new_instance_id)
                report[new_instance_id]["timed_out"] = timed_out
                report[new_instance_id]["success"] = success
                
            except Exception as e:
                report = create_error_report(new_instance_id, str(e))
                    
            end_time = time.time()
            duration = end_time - start_time
            report[new_instance_id]["duration"] = duration
            
    except Exception as e:
        logger.error(f"Error evaluating init instance {new_instance_id}: {str(e)}")
        report = create_error_report(new_instance_id, str(e))
    
    return report

def test_gpt_bug(tasks, max_workers, timeout):
    """测试GPT生成的bug"""
    log_path = f"{EXP_PATH}/gpt_bug_eval_log"
    os.makedirs(log_path, exist_ok=True)
    
    start_time = time.time()
    results = run_parallel_tasks(tasks, log_path, eval_gpt_bug_instance, max_workers=max_workers, timeout=timeout)
    total_time = time.time() - start_time
    
    logger.info("\nGPT Bug Evaluation Summary:")
    logger.info(f"Total time: {total_time:.2f}s")
    if results:
        logger.info(f"Average task time: {total_time/len(results):.2f}s")
    
    # 保存所有结果
    gpt_bug_results_path = f"{EXP_PATH}/gpt_bug_all_results.json"
    save_evaluation_results(results, gpt_bug_results_path, "GPT bug")
    
    return results

def eval_gpt_bug_instance(instance: dict, log_path: str, timeout=100) -> dict:
    """评估GPT生成的bug实例"""
    instance_id = instance['new_instance_id']
    # new_instance_id = instance['new_instance_id']
    repo = instance['repo']
    base_commit = instance['base_commit']
    repo_commit = get_repo_commit_name(repo, base_commit)
    
    # 获取GPT生成的patches
    gpt_patch = instance.get('gpt_patch', '')  # GPT生成的bug patch
    
    if not gpt_patch:
        return create_error_report(instance_id, "Missing gpt_patch")
    
    # 创建临时patch文件
    gpt_bug_patch_path = create_temp_patch_file(gpt_patch, GENERATE_DATA_PATH, instance_id, "gpt-bug")
    init_test_patch_path = create_temp_patch_file(instance['test_patch'], GENERATE_DATA_PATH, instance_id, "gpt-init")
    
    # 获取需要复制的文件
    gpt_patch_files = get_all_filenames(gpt_patch)
    files_to_copy = merge_files_to_copy([gpt_patch_files])
    
    source_testbed = os.path.join(DEFAULT_PATH, repo_commit)
    conda_path = os.path.join(ENV_DIR, repo_commit)
    eval_sh = "./eval.sh"
    
    # 检查脚本是否存在
    if not os.path.exists(eval_sh):
        return create_error_report(instance_id, f"Evaluation script not found: {eval_sh}")
    
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
                return create_error_report(instance_id, f"Failed to apply GPT bug patch: {result_2.stderr}")

            # 运行测试
            cmd = f'bash {eval_sh} {temp_dir} "{test_command}" {init_test_patch_path}'
            cmd += f' > {instance_log} 2>&1'
            
            start_time = time.time()
            try:
                # 运行命令并获取结果
                success, timed_out = run_command_with_timeout(
                    instance_id, cmd, timeout=timeout
                )
                
                report = get_eval_report_synthetic(instance, instance_log, True, instance_id)
                report[instance_id]["timed_out"] = timed_out
                report[instance_id]["success"] = success
                
            except Exception as e:
                report = create_error_report(instance_id, str(e))
                    
            end_time = time.time()
            duration = end_time - start_time
            report[instance_id]["duration"] = duration
            
    except Exception as e:
        logger.error(f"Error evaluating GPT bug instance {instance_id}: {str(e)}")
        report = create_error_report(instance_id, str(e))
    
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

def filter_tasks_by_test_count(tasks: List[dict], init_results: dict, min_passed_tests: int = 5) -> Tuple[List[dict], dict]:
    """
    根据init测试结果筛选有足够通过测试的任务,比如10个任务，只有6个满足，则只保留满足条件的数据进行输出
    
    Args:
        tasks: 任务列表
        init_results: init测试结果
        min_passed_tests: 最少通过测试数量
        
    Returns:
        Tuple[List[dict], dict]: (筛选后的任务列表, 筛选后的init结果)
    """
    logger.info(f"Filtering tasks with at least {min_passed_tests} passing tests...")
    filtered_tasks = []
    filtered_init_results = {}
    
    for task in tasks:
        instance_id = task['new_instance_id']
        if instance_id in init_results:
            init_result = init_results[instance_id]
            passed_tests = init_result.get('tests_status', {}).get('PASSED', [])
            passed_count = len(passed_tests)
            
            if passed_count >= min_passed_tests:
                filtered_tasks.append(task)
                filtered_init_results[instance_id] = init_result
                logger.debug(f"Task {instance_id} kept with {passed_count} passing tests")
            else:
                logger.info(f"Task {instance_id} filtered out with only {passed_count} passing tests")
        else:
            logger.warning(f"Task {instance_id} not found in init results, skipping")
    
    logger.info(f"Filtering complete: {len(filtered_tasks)}/{len(tasks)} tasks kept")
    return filtered_tasks, filtered_init_results

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
            # Check if task has 'response' field (from regenerated prompt) or 'meta_response' (original)
            if 'response' in task:
                # Use the regenerated response
                response = task['response']
            elif 'meta_response' in task:
                # Use the original meta_response (fallback for compatibility)
                meta_response = json.loads(task.get('meta_response'))
                response = meta_response['choices'][0]['message']['content']
            else:
                logger.warning(f"Skipping {instance_id}: No response or meta_response found.")
                failed_ids.add(instance_id)
                continue
            
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

def check_and_retry_insufficient_tests(
    tasks_to_evaluate: List[dict], 
    threshold: int = 5, 
    max_retries: int = 5,
    load_history: bool = False
) -> Tuple[List[dict], Dict[str, dict]]:
    """
    检查并重试测试不足的任务，包含完整的test_init调用和重试循环
    
    Args:
        tasks_to_evaluate: 要评估的任务列表
        threshold: 通过测试的最小数量阈值
        max_retries: 最大重试次数
        load_history: 是否直接读取历史信息，如果为True则直接返回历史信息
    
    Returns:
        Tuple[List[dict], Dict[str, dict]]: (final_tasks, all_init_results)
    """
    
    # 历史文件路径
    history_file = f"{GENERATE_DATA_PATH}/check_and_retry_history.json"
    
    # 如果load_history为True，尝试读取历史信息
    if load_history:
        logger.info("Loading history information...")
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                logger.info("Successfully loaded history information")
                
                # 检查历史中的iteration次数
                current_iteration = history_data.get('iteration', 0)
                logger.info(f"History shows {current_iteration} iterations completed")
                
                # 如果历史中的iteration次数已达到max_retries，直接返回历史结果
                if current_iteration >= max_retries:
                    logger.info(f"History iteration ({current_iteration}) >= max_retries ({max_retries}), returning history results")
                    return history_data.get('final_tasks', []), history_data.get('all_init_results', {})
                else:
                    logger.info(f"History iteration ({current_iteration}) < max_retries ({max_retries}), continuing from history")
                    # 从历史数据继续，但不直接返回
                    # 设置起始状态为历史数据
                    tasks_to_evaluate = history_data.get('final_tasks', tasks_to_evaluate)
                    retry_count = current_iteration
                    all_init_results = history_data.get('all_init_results', {})
                    
                    # 重新运行最后一次的init测试以获取当前状态
                    current_init_results = test_init(tasks_to_evaluate, max_workers=100, timeout=50)
                    current_tasks = tasks_to_evaluate.copy()
                    all_sufficient_tasks = []
                    
                    # 将当前结果更新到全局结果中
                    all_init_results.update(current_init_results)
                    
                    # 跳过初始化部分，直接进入循环
                    logger.info(f"Continuing from history with retry_count={retry_count}")
                    # 使用goto到循环开始处的逻辑
                    # 这里我们将通过设置变量来实现继续逻辑
                    continue_from_history = True
            else:
                logger.warning("History file not found, starting fresh evaluation")
                continue_from_history = False
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}, starting fresh evaluation")
            continue_from_history = False
    else:
        continue_from_history = False
    
    # 如果不是从历史继续，执行正常的初始化
    if not continue_from_history:
        logger.info(f"Starting test evaluation with up to {max_retries} retries...")
        
        # 初始运行test_init获取基准结果
        current_init_results = test_init(tasks_to_evaluate, max_workers=CONC, timeout=100)
        current_tasks = tasks_to_evaluate.copy()
        retry_count = 0
        all_sufficient_tasks = []
        
        # 用于存储所有任务的完整init结果
        all_init_results = {}
        
        # 将初始结果添加到全局结果中
        all_init_results.update(current_init_results)
        
        # 存储初始状态
        history_data = {
            'final_tasks': [],
            'all_init_results': all_init_results.copy(),
            'iteration': 0
        }
        
        # 保存初始状态
        try:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save initial history: {str(e)}")
    
    while retry_count < max_retries and current_tasks:
        current_sufficient = []
        tasks_need_retry = []
        
        for task in current_tasks:
            instance_id = task['new_instance_id']
            if instance_id in current_init_results:
                init_result = current_init_results[instance_id]
                passed_tests = init_result.get('tests_status', {}).get('PASSED', [])
                passed_count = len(passed_tests)
                
                if passed_count < threshold:
                    tasks_need_retry.append(task)
                else:
                    current_sufficient.append(task)
            else:
                logger.warning(f"Task {instance_id} not found in init results, adding to retry")
                tasks_need_retry.append(task)
        
        # 将本轮通过的任务添加到累积列表
        all_sufficient_tasks.extend(current_sufficient)
        logger.info(f"Current iteration: {len(current_sufficient)} tasks passed, total accumulated: {len(all_sufficient_tasks)}")
        
        if not tasks_need_retry:
            logger.info("All remaining tasks have sufficient test coverage, no more retry needed")
            break
            
        retry_count += 1
        logger.info(f"Retry attempt {retry_count}/{max_retries}: Retrying {len(tasks_need_retry)} insufficient tasks...")
        
        # 重新生成任务
        retried_tasks = retry_tasks_in_parallel(tasks_need_retry, DEFAULT_PATH)
        
        # 对重新生成的任务执行test_init
        retry_init_results = test_init(retried_tasks, max_workers=CONC, timeout=100)
        
        # 更新当前任务和测试结果
        current_tasks = retried_tasks
        current_init_results = retry_init_results
        
        # 将重试结果也添加到全局结果中
        all_init_results.update(retry_init_results)
        
        # 合并当前结果
        final_tasks = all_sufficient_tasks + current_tasks
        
        # 在每次循环中存储或更新final_tasks和all_init_results
        history_data = {
            'final_tasks': final_tasks,
            'all_init_results': all_init_results.copy(),
            'iteration': retry_count
        }
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
            logger.info(f"Updated history file with iteration {retry_count} results")
        except Exception as e:
            logger.warning(f"Failed to update history: {str(e)}")
        
        logger.info(f"Retry {retry_count} phase completed. Tests re-evaluated for {len(retried_tasks)} tasks")
    
    # 最终结果
    final_tasks = all_sufficient_tasks + current_tasks
    logger.info(f"Test evaluation with retries completed. Total sufficient tasks: {len(all_sufficient_tasks)}, remaining tasks after max retries: {len(current_tasks)}, total: {len(final_tasks)}")
    
    # 存储最终结果
    history_data = {
        'final_tasks': final_tasks,
        'all_init_results': all_init_results.copy(),
        'iteration': retry_count
    }
    
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2)
        logger.info("Final results saved to history file")
    except Exception as e:
        logger.warning(f"Failed to save final history: {str(e)}")
    
    return final_tasks, all_init_results


def check_and_retry_buggy_tests(
    tasks_to_evaluate: List[dict], 
    init_results: dict,
    max_retries: int = 10,
    load_history: bool = True
) -> Tuple[List[dict], dict]:
    """
    Check and retry buggy tests with insufficient bug detection using multi-round retry mechanism.
    
    Args:
        tasks_to_evaluate: List of tasks to evaluate
        init_results: Initial test results for each task
        max_retries: Maximum number of retry attempts
        load_history: Whether to load history from previous runs
        
    Returns:
        Tuple of (final_tasks, final_bug_results) after retries
    """
    current_tasks = tasks_to_evaluate.copy()
    all_bug_results = {}
    retry_count = 0
    
    # Categories to retry
    retry_categories = ['bug_not_detected', 'other_cases']
    
    # Load history if available
    history_file = os.path.join(HISTORY_DIR, "buggy_retry_history.json")
    history_data = None
    
    if load_history and os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            logger.info(f"Loaded retry history from {history_file}")
            if 'final_bug_results' in history_data:
                all_bug_results.update(history_data['final_bug_results'])
                logger.info(f"Loaded {len(all_bug_results)} existing bug results from history")
        except Exception as e:
            logger.warning(f"Failed to load retry history: {str(e)}")
    
    while retry_count < max_retries and current_tasks:
        logger.info(f"=== Buggy Test Retry Iteration {retry_count + 1}/{max_retries} ===")
        logger.info(f"Processing {len(current_tasks)} tasks...")
        
        # Run GPT bug evaluation
        logger.info("Running GPT bug evaluation...")
        bug_results = test_gpt_bug(current_tasks, max_workers=CONC, timeout=100)
        
        # Merge new results with existing ones
        all_bug_results.update(bug_results)
        
        # Analyze combined results
        analysis_results = analyze_combined_results(init_results, all_bug_results)
        
        # Collect tasks that need retry
        tasks_needing_retry = []
        for category in retry_categories:
            for instance_id, result_data in analysis_results[category].items():
                # Find the original task
                task = next((t for t in tasks_to_evaluate if t['instance_id'] == instance_id), None)
                if task and task in current_tasks:
                    tasks_needing_retry.append(task)
        
        if not tasks_needing_retry:
            logger.info("All tasks have sufficient bug detection. Stopping retries.")
            break
            
        logger.info(f"Found {len(tasks_needing_retry)} tasks needing retry")
        
        # Retry the buggy code generation for these tasks
        logger.info("Retrying buggy code generation...")
        retried_tasks = retry_buggy_code_in_parallel(
            tasks_needing_retry, 
            max_workers=CONC,
            retry_count=retry_count + 1  # Pass retry count for better logging
        )
        
        # Prepare for next iteration
        current_tasks = retried_tasks
        retry_count += 1
        
        # Save progress
        history_data = {
            'final_tasks': current_tasks,
            'final_bug_results': all_bug_results.copy(),
            'iteration': retry_count
        }
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
            logger.info(f"Retry progress saved to {history_file}")
        except Exception as e:
            logger.warning(f"Failed to save retry progress: {str(e)}")
    
    # Final analysis after all retries
    final_analysis = analyze_combined_results(init_results, all_bug_results)
    
    # Filter to only include tasks that were originally provided
    final_tasks = [t for t in tasks_to_evaluate if t['instance_id'] in all_bug_results]
    final_bug_results = {k: v for k, v in all_bug_results.items() 
                        if k in [t['instance_id'] for t in final_tasks]}
    
    logger.info(f"Buggy test evaluation completed. Total tasks: {len(final_tasks)}, "
                f"retry iterations: {retry_count}")
    
    # Save final results
    history_data = {
        'final_tasks': final_tasks,
        'final_bug_results': final_bug_results,
        'iteration': retry_count
    }
    
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2)
        logger.info("Final buggy test results saved to history file")
    except Exception as e:
        logger.warning(f"Failed to save final buggy test history: {str(e)}")
    
    return final_tasks, final_bug_results



def run_evaluation_phase(tasks_to_evaluate: List[dict], load_history: bool):
    """
    Runs the full evaluation pipeline (init, gpt_bug) and analysis.
    
    Returns:
        The dictionary of analysis results.
    """
    logger.info("=== Phase 3.1 === Starting construct init state evaluation...")
    
    # 检查并重试不足的任务（同时获取最终测试结果）
    final_tasks, final_init_results = check_and_retry_insufficient_tests(tasks_to_evaluate, threshold=5, max_retries=10, load_history=load_history)
    
    # # 根据最终init结果筛选有足够通过测试的任务，把不满足的删除
    filtered_final_tasks, filtered_final_init_results = filter_tasks_by_test_count(final_tasks, final_init_results, 5)
    
    logger.info("=== Phase 3.2 === Starting GPT bug evaluation ...")

    # 使用多轮重试机制检查并重试bug检测不足的任务
    final_tasks, final_bug_results = check_and_retry_buggy_tests(
        filtered_final_tasks, 
        filtered_final_init_results, 
        max_retries=10, 
        load_history=load_history
    )
    
    # 分析最终结果
    logger.info("Analyzing final combined results...")
    analysis_results = analyze_combined_results(filtered_final_init_results, final_bug_results)

    return analysis_results

# ================== Main Execution ==================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract bugs and evaluate GPT bug detection with retry mechanism.')

    parser.add_argument('--max_retries', type=int, default=5, 
                        help='Maximum number of retries for failed items (default: 5). Set to 1 to disable retries.')
    parser.add_argument('--enable_retry', action='store_true', 
                        help='Enable the retry mechanism for failed items. If not set, runs only once.')
    parser.add_argument('--restart', action='store_true',
                        help='Restart from saved checkpoint if available')
    parser.add_argument('--input_jsonl', type=str, default='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data/script_ranker.jsonl',
                        help='Path to input JSONL file containing instance_id mapping')
    parser.add_argument('--output_jsonl', type=str, default=f"{GENERATE_DATA_PATH}/task2_multiscript/gpt_2_finish_bug_gpt4o_ranker.jsonl",
                        help='Path to output JSONL file')
    parser.add_argument('--load_history', type=bool, default=True, 
                        help='whether load unittest test results from history')

    args = parser.parse_args()
    
    start_time = time.time()

    # --- Define intput and output file ---
    final_output_file = args.output_jsonl
    input_jsonl_file = args.input_jsonl
    # --- Load Initial Data ---
    # Always load initial data from the source file
    initial_tasks = []    
    # if not os.path.exists(input_jsonl_file):
    #     print(f"Warning: Input file not found: {input_jsonl_file}. Skipping.")
    #     input_jsonl_file = input_jsonl_file.replace(".tmp", "")
    try:
        with open(input_jsonl_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                if 'ranker_info' in data and data['ranker_info']:
                    # 使用 enumerate 遍历 ranker_info，从 1 开始计数
                    for i, (file_name, meta_data) in enumerate(data['ranker_info'].items(), start=1):
                        new_data = data.copy()
                        random.seed(data['instance_id']+file_name)
                        new_data['new_base_commit'] = ''.join(random.choices('0123456789abcdef', k=40))
                        new_data['new_instance_id'] = new_data['repo'].replace("/", "__")+"__"+ new_data['new_base_commit'][:6]
                        new_data['main_script'] = file_name
                        new_data['main_script_metadata'] = meta_data
                        new_data['main_script_metadata']['script_rank'] = i
                        new_data.pop('ranker_info')
                        initial_tasks.append(new_data)
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
        tasks_to_process = tasks_to_process
        logger.info(f"Restarting with fresh attempt counting (max_retries: {args.max_retries})")
        logger.info(f"Already processed: {len(processed_ids)} tasks")
        logger.info(f"Remaining to process: {len(tasks_to_process)} tasks")
    else:
        # Normal initialization
        tasks_to_process = initial_tasks[:TEST_N]
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

        # For ALL attempts, regenerate the LLM response using prompt reconstruction
        logger.info(f"Generating LLM responses for attempt {attempt}...")
        
        # Use ThreadPoolExecutor for concurrent requests with progress bar
        max_workers = min(CONC, len(tasks_to_process))
        if not args.load_history:
            print('Phase 1: Generate first round response')
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks with reconstruction
                future_to_task = {
                    executor.submit(process_single_task_with_reconstruction, task, all_perfect_tasks, tasks_to_process, DEFAULT_PATH): task
                    for task in tasks_to_process
                }
                
                # Process completed tasks with progress bar
                completed_tasks = []
                desc = f"Generating LLM responses (attempt {attempt})"
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
            
            # Phase 2: Extract Patches from LLM responses
            print('Phase 2: Extract Patches from LLM responses')
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
        else:
            logger.info('Load history from file.')
            extracted_tasks = None
        # Phase 3: Evaluate successfully extracted patches
        print('Phase 3: Evaluate successfully extracted patches')
        analysis_results = run_evaluation_phase(extracted_tasks, args.load_history)
        
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
            task for task in initial_tasks[:TEST_N]
            if task['instance_id'] not in processed_ids
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