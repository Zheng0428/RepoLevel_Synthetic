import subprocess
import threading
import time
import signal
import os
from grading_simple import get_eval_report, get_eval_report_synthetic
import sys
import json
import re
import tempfile
from datasets import load_dataset
from pathlib import Path
import logging
from temp_testbed import TempTestbed, get_all_filenames
from envs import DEFAULT_PATH

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union, Tuple

from utils import fake_git_repo
from envs import GENERATE_DATA_PATH, LOG_PATH

# Import functions from 3_2_extract_goodtasks.py
import sys

# Add the path to 'another_folder' to the system path
from utils import _filter_test_content, _generate_combined_test_script

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - Miscellaneous
date = time.strftime("%Y-%m-%d-%H")

EXP_PATH=f"{LOG_PATH}/test_log/py-gpt-bug-patch-commit_{date}_combined"
    
EXPR_PATH = os.getenv("EXPR_PATH", "/opt/tiger/expr")
ENV_DIR=f"{EXPR_PATH}/conda_env/"
# REPO_DIR=f"{EXPR_PATH}/repo_commit/"

NON_TEST_EXTS = [
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
]


def run_command_with_timeout(instance_id, cmd, timeout) -> tuple[bool, bool]:  
    """  
    执行命令并支持超时终止，返回执行成功状态和是否超时标志  
    
    参数:  
        instance_id: 实例标识符(用于日志或跟踪)  
        cmd: 要执行的命令  
        timeout: 超时时间(秒)  
        
    返回:  
        (success, timed_out):   
            - success: 命令是否成功执行(True/False)  
            - timed_out: 是否因超时而终止(True/False)  
    """  
    # 创建新进程组  
    process = subprocess.Popen(  
        cmd,  
        shell=True,  
        text=True,  
        preexec_fn=os.setsid  # 创建新会话，使进程成为进程组长  
    )  
    
    try:  
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode == 0:  
            return True, False  
        else:  
            logger.error(f"Task {instance_id} failed with return code {process.returncode}")  
            if stderr:
                logger.error(f"stderr: {stderr}")  
            return False, False  
    except subprocess.TimeoutExpired:  
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  
        time.sleep(1)  
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)  
        logger.error(f"Task {instance_id} timed out")
        return False, True
    except Exception as e:  
        # 处理其他可能的异常  
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)  
        logger.error(f"Unexpected error in task {instance_id}: {str(e)}")
        return False, False


def make_test_command_for_gpt(instance, env_dir, tmp_dir, test_patch) -> str:
    """为GPT生成的测试创建测试命令"""
    pytest_path = os.path.join(env_dir, "bin", "pytest")
    test_cmd = (
        f"{pytest_path} --no-header -rA "
        f"-p no:cacheprovider "
        f"--basetemp={tmp_dir} "
        f"-W ignore::DeprecationWarning"
    )
    
    # 获取GPT测试patch中的文件指令
    diff_pat = r"diff --git a/.* b/(.*)"
    # test_patch = test_patch
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if instance['repo'] == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    test_command = " ".join([test_cmd, *directives])  
    return test_command


##################################################################################################################################



def test_init(tasks, max_workers, timeout):
    """测试初始状态（修复后）GPT生成的测试是否能通过"""
    log_path = f"{EXP_PATH}/init_eval_log"
    os.makedirs(log_path, exist_ok=True)
    
    start_time = time.time()
    results = eval_parallel_init_tasks(tasks, log_path, max_workers=max_workers, timeout=timeout)
    total_time = time.time() - start_time
    
    logger.info("\nInit State Evaluation Summary:")
    logger.info(f"Total time: {total_time:.2f}s")
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
    
    Args:
        instance: 包含GPT生成的patch信息的字典
        log_path: 日志路径
        timeout: 超时时间
        
    Returns:
        dict: 评估报告
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
    # original_patch_path = f"{init_test_dir}/original_patch.diff"
    
    # 创建目录
    os.makedirs(os.path.dirname(init_test_patch_path), exist_ok=True)
    
    # 写入patch文件
    with open(init_test_patch_path, 'w') as f:
        f.write(test_patch)
        
    # with open(original_patch_path, 'w') as f:
    #     f.write(original_patch)
    
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
            test_command = make_test_command_for_gpt(instance, conda_path, temp_pytest, test_patch)
            
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
                        "timed_out": False
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




##################################################################################################################################


def test_gpt_bug(tasks, max_workers, timeout):
    """测试GPT生成的bug"""
    log_path = f"{EXP_PATH}/gpt_bug_eval_log"
    os.makedirs(log_path, exist_ok=True)
    
    start_time = time.time()
    results = eval_parallel_gpt_bug_tasks(tasks, log_path, max_workers=max_workers, timeout=timeout)
    total_time = time.time() - start_time
    
    logger.info("\nGPT Bug Evaluation Summary:")
    logger.info(f"Total time: {total_time:.2f}s")
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
    """
    评估GPT生成的bug实例
    
    Args:
        instance: 包含GPT生成的patch信息的字典
        log_path: 日志路径
        timeout: 超时时间
        
    Returns:
        dict: 评估报告
    """
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
                "error": "Missing gpt_patch or gpt_test_patch",
                "timed_out": False
            }
        }
    
    # 创建临时patch文件路径
    gpt_bug_patch_path = f"{GENERATE_DATA_PATH}/gpt-bug-eval/{instance_id}/gpt_bug_patch.diff"
    init_test_patch_path = f"{GENERATE_DATA_PATH}/gpt-init-eval/{instance_id}/test_patch.diff"
    # original_patch_path = f"/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/eval_task_commit/gpt-bug-eval/{instance_id}/original_patch.diff"
    
    # 创建目录
    os.makedirs(os.path.dirname(gpt_bug_patch_path), exist_ok=True)
    
    # 写入patch文件
    with open(gpt_bug_patch_path, 'w') as f:
        f.write(gpt_patch)
        
    # with open(original_patch_path, 'w') as f:
    #     f.write(original_patch)
    
    # 获取需要复制的文件
    # original_patch_files = get_all_filenames(original_patch)
    gpt_patch_files = get_all_filenames(gpt_patch)
    
    # 合并所有需要的文件
    files_to_copy = list(set(
        # original_patch_files["modified"] + 
        # original_patch_files["added"] +
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
            test_command = make_test_command_for_gpt(instance, conda_path, temp_pytest, instance['test_patch'])
            
            # 确保使用绝对路径
            log_path = os.path.abspath(log_path)
            os.makedirs(log_path, exist_ok=True)
            instance_log = os.path.join(log_path, f"{instance_id}.log")
            
            # 在临时目录中先应用原始patch，再应用逆向的gpt_patch，最后应用test_patch
            # 由于我们需要特殊的顺序处理，我们需要手动处理patch应用
            
            # 步骤1: 应用原始patch (修复bug)
            # patch_cmd_1 = f'cd {temp_dir} && git apply --whitespace=nowarn {original_patch_path}'
            # result_1 = subprocess.run(patch_cmd_1, shell=True, capture_output=True, text=True)
            # if result_1.returncode != 0:
            #     return {
            #         instance_id: {
            #             "success": False,
            #             "error": f"Failed to apply original patch: {result_1.stderr}",
            #             "timed_out": False,
            #             "duration": 0
            #         }
            #     }
            
            # 步骤2: 应用GPT bug patch (逆向，引入bug)
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

            # 步骤3: 运行测试
            cmd = f'bash {eval_sh} {temp_dir} "{test_command}" {init_test_patch_path}'

            cmd += f' > {instance_log} 2>&1'


            # cmd = f'cd {temp_dir} && {test_command} > {instance_log} 2>&1'
            
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
                        "timed_out": False
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


##################################################################################################################################


def extract_perfect_tests_unittest(perfect_tests_results: dict, original_data: dict) -> dict:
    """
    从perfect_tests结果中提取测试，生成完整的unittest代码
    
    Args:
        perfect_tests_results: analyze_combined_results返回的perfect_tests结果
        original_data: 原始数据，包含gpt_test_patch信息
        
    Returns:
        dict: 包含instance_id到unittest代码的映射
    """
    unittest_results = {}
    
    for instance_id, result_data in perfect_tests_results.items():
        perfect_tests_list = result_data.get('perfect_tests', [])
        
        if not perfect_tests_list:
            continue
        
        # 从original_data中获取对应的实例数据
        instance_data = None
        for data in original_data:
            if data.get('instance_id') == instance_id:
                instance_data = data
                break
        
        if not instance_data:
            logger.warning(f"Could not find original data for instance {instance_id}")
            continue
        
        gpt_test_patch = instance_data.get('gpt_test_patch', '')
        if not gpt_test_patch:
            logger.warning(f"No gpt_test_patch found for instance {instance_id}")
            continue
        
        # 解析gpt_test_patch以获取测试文件内容
        test_files = parse_test_patch_content(gpt_test_patch)
        
        if not test_files:
            logger.warning(f"Could not parse test files from gpt_test_patch for instance {instance_id}")
            continue
        
        # 为每个perfect test创建过滤后的测试内容
        good_tests = {}
        
        for test_file_path, test_content in test_files.items():
            # 为当前文件找到相关的perfect tests
            relevant_perfect_tests = []
            for perfect_test in perfect_tests_list:
                # perfect_test格式可能是 "test_file.py::TestClass::test_method" 或 "test_file.py::test_function"
                if perfect_test.startswith(test_file_path.replace('.py', '').replace('/', '__')):
                    relevant_perfect_tests.append(perfect_test)
            
            if not relevant_perfect_tests:
                continue
            
            # 转换perfect_tests格式为_filter_test_content需要的格式
            class_function_specs = []
            for perfect_test in relevant_perfect_tests:
                parts = perfect_test.split('::')
                if len(parts) >= 2:
                    # 去除文件名部分，只保留类::方法 或 ::方法
                    if len(parts) == 2:
                        # 格式: file::function
                        class_function_specs.append('::' + parts[1])
                    elif len(parts) == 3:
                        # 格式: file::class::method
                        class_function_specs.append(parts[1] + '::' + parts[2])
            
            if class_function_specs:
                # 过滤测试内容，只保留perfect tests
                filtered_content = _filter_test_content(test_content, class_function_specs)
                good_tests[test_file_path] = {
                    'content': filtered_content,
                    'class::function': class_function_specs
                }
        
        if good_tests:
            # 生成合并的测试脚本
            if len(good_tests) == 1:
                combined_unittest = next(iter(good_tests.values()))['content']
            else:
                combined_unittest = _generate_combined_test_script(good_tests)
            
            unittest_results[instance_id] = {
                'gpt_final_unittest': combined_unittest,
                'perfect_tests': perfect_tests_list,
                'test_files_used': list(good_tests.keys())
            }
    
    return unittest_results


def parse_test_patch_content(gpt_test_patch: str) -> dict:
    """
    解析gpt_test_patch，提取测试文件的内容
    
    Args:
        gpt_test_patch: GPT生成的测试patch内容
        
    Returns:
        dict: 文件路径到内容的映射
    """
    test_files = {}
    
    # 分割patch内容，按文件处理
    file_sections = re.split(r'^diff --git', gpt_test_patch, flags=re.MULTILINE)
    
    for section in file_sections:
        if not section.strip():
            continue
        
        # 如果不是以diff --git开头，添加回去（除了第一个section）
        if not section.startswith(' a/') and not section.startswith('diff --git'):
            section = 'diff --git ' + section
        
        # 提取文件路径 - 修复正则表达式
        file_match = re.search(r'diff --git a/(.+) b/(.+)', section)
        if not file_match:
            # 尝试另一种格式：直接从文件名开始
            file_match = re.search(r'^(.+\.py) b/(.+\.py)', section)
        
        if not file_match:
            continue
        
        file_path = file_match.group(2) if file_match.group(2) else file_match.group(1)
        
        # 只处理.py文件
        if not file_path.endswith('.py'):
            continue
        
        # 提取文件内容
        lines = section.split('\n')
        content_lines = []
        in_content = False
        
        for line in lines:
            # 检查是否到达内容区域
            if line.startswith('@@') and '-0,0' in line and '+1,' in line:
                # 这是新文件的标记
                in_content = True
                continue
            elif line.startswith('@@'):
                in_content = True
                continue
            elif line.startswith('+++') and '/dev/null' not in line:
                # 跳过文件头信息
                continue
            elif line.startswith('---') or line.startswith('index ') or line.startswith('new file mode'):
                # 跳过文件头信息
                continue
            
            if in_content:
                if line.startswith('+') and not line.startswith('+++'):
                    # 移除开头的+号，这是新增的内容
                    content_lines.append(line[1:])
                elif line.startswith(' ') and len(line) > 1:
                    # 上下文行（空格开头）
                    content_lines.append(line[1:])
                elif line.startswith('-'):
                    # 删除的行，忽略
                    continue
                elif not line.startswith(('+', '-', ' ', '\\', '@')):
                    # 其他行也可能是内容
                    content_lines.append(line)
        
        if content_lines:
            # 清理内容：移除开头和结尾的空行
            while content_lines and not content_lines[0].strip():
                content_lines.pop(0)
            while content_lines and not content_lines[-1].strip():
                content_lines.pop()
            
            if content_lines:
                test_files[file_path] = '\n'.join(content_lines)
    
    return test_files


def save_final_results_to_jsonl(perfect_tests_results: dict, original_data: dict, output_file: str):
    """
    保存最终结果到jsonl文件
    
    Args:
        perfect_tests_results: perfect_tests结果
        original_data: 原始数据列表
        output_file: 输出文件路径
    """
    # 提取unittest代码
    unittest_results = extract_perfect_tests_unittest(perfect_tests_results, original_data)
    
    # 创建输出数据
    output_data = []
    
    for data in original_data:
        instance_id = data.get('instance_id')
        
        if instance_id in unittest_results:
            # 复制原始数据
            new_data = data.copy()
            
            # 添加生成的unittest信息
            unittest_info = unittest_results[instance_id]
            gpt_valid_test_patch = fake_git_repo(
                file_pathes=unittest_info['test_files_used'],
                old_contents=[""],
                new_contents=[unittest_info['gpt_final_unittest']],
                files_exist=False
            )
            new_data['gpt_valid_test_patch'] = gpt_valid_test_patch
            new_data['gpt_perfect_tests'] = unittest_info['perfect_tests']
            
            output_data.append(new_data)
    
    # 保存到jsonl文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for data in output_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(output_data)} final results to {output_file}")
    return output_data


def analyze_combined_results(init_results, gpt_bug_results):
    """
    分析合并的结果，筛选出符合条件的测试
    条件：至少有一个test在init状态通过 && 在gpt_bug状态失败
    """
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
    
    logger.info("Analysis results saved:")
    logger.info(f"  Perfect tests: {perfect_tests_path}")
    logger.info(f"  Init failed: {init_failed_path}")
    logger.info(f"  Bug not detected: {bug_not_detected_path}")
    logger.info(f"  Other cases: {other_cases_path}")
    logger.info(f"  Final comprehensive results: {final_results_path}")
    
    return {
        'perfect_tests': good_tests,
        'init_failed': init_failed,
        'bug_not_detected': bug_not_detected,
        'other_cases': other_cases,
        'summary': final_results['summary']
    }


# 使用示例：
if __name__ == "__main__":
    start_time = time.time()
    
    # 从extract_bug_gpt.py的输出文件中读取数据
    data_path = f'{GENERATE_DATA_PATH}/gpt_1_bug_gpt4o.jsonl'
    tasks = []
    
    try:
        with open(data_path, 'r') as f_in:
            for line in f_in:
                task_data = json.loads(line)
                # 只处理有GPT patch的数据
                if 'gpt_patch' in task_data:
                    tasks.append(task_data)
                else:
                    logger.warning(f"Skipping task {task_data.get('instance_id', 'unknown')}: missing GPT patches")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(tasks)} tasks with GPT patches")
    if tasks:
        # 先运行test_init，检查修复状态下GPT测试是否能通过
        logger.info("Starting init state evaluation...")
        init_results = test_init(tasks, 32, 45)
        
        # 再运行test_gpt_bug，检查引入bug后GPT测试是否能检测到bug
        logger.info("Starting GPT bug evaluation...")
        gpt_bug_results = test_gpt_bug(tasks, 32, 45)
        
        # 分析合并结果
        logger.info("Analyzing combined results...")
        analysis_results = analyze_combined_results(init_results, gpt_bug_results)

        ## test
        # EXP_PATH = '/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/test_log/py-gpt-bug-patch-commit_2025-06-16-12_combined'
        # final_results_path = f"{EXP_PATH}/final_analysis_results.json"
        # with open(final_results_path, "r") as f:
        #     analysis_results = json.load(f)
        # analysis_results = analysis_results['categorized_results']

        
        # 新增：生成最终的unittest代码并保存到jsonl
        logger.info("Generating final unittest code...")
        final_output_file = f"{GENERATE_DATA_PATH}/7_finish_bug_gpt4o.jsonl"
        final_data = save_final_results_to_jsonl(
            analysis_results['perfect_tests'], 
            tasks, 
            final_output_file
        )
        
        logger.info(f"Evaluation and analysis complete. Results saved to {EXP_PATH}")
        logger.info(f"Perfect tests found: {len(analysis_results['perfect_tests'])}")
        logger.info(f"Final unittest data saved to: {final_output_file}")
        logger.info(f"Final data count: {len(final_data)}")
        
    else:
        logger.warning("No valid tasks found to evaluate")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f}s")
