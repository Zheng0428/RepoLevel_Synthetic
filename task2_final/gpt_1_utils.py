import json
import re
import os
import logging
import subprocess
import signal
import time
import tempfile
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from temp_testbed import TempTestbed, get_all_filenames
from utils import fake_git_repo
from utils import get_llm_response as get_model_resposne #get_llm_response, get_deepseek_response
from envs import DEFAULT_PATH, TRUE_PROJECT_FILE_LOC
from utils import construct_three_shot_prompt as construct_prompt
from utils import construct_unittest_prompt as construct_unittest_prompt
from utils import construct_buggy_prompt as construct_buggy_prompt
CONC=2
TEST_N=20



logger = logging.getLogger(__name__)

# Constants
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

@dataclass
class BugFile:
    """表示一个包含bug的文件"""
    file_path: str
    code: str

@dataclass
class UnittestFile:
    """表示一个unittest文件"""
    file_path: str
    code: str

@dataclass
class ParsedBugResponse:
    """解析后的bug响应结构"""
    problem_statement: str
    bug_analysis: str
    buggy_files: List[BugFile]
    unittest_file: Optional[UnittestFile]  # 新增unittest文件字段

# ================== Text Processing Functions ==================

def clean_text(text):
    """Remove special characters from the beginning and end, including spaces, newline characters, asterisks, quotes, and colons."""
    return re.sub(r'^[\s\*\n\'\"""''：:]+|[\s\*\n\'\"""''：:]+$', '', text)

# ================== Response Parsing Functions ==================

def parse_bug_response(response: str) -> Optional[ParsedBugResponse]:
    """
    解析符合 three_shot_same_test.yaml 格式的响应
    
    Args:
        response: 需要解析的响应字符串
        
    Returns:
        ParsedBugResponse: 解析后的结构化数据，如果解析失败返回None
    """
    try:
        # 解析问题陈述
        problem_statement = _extract_section(response, "PROBLEM_STATEMENT")
        if not problem_statement:
            return None
            
        # 解析bug分析
        bug_analysis = _extract_section(response, "BUG_ANALYSIS")
        if not bug_analysis:
            return None
            
        # 解析buggy文件
        buggy_files = _extract_buggy_files(response)
        if not buggy_files:
            return None

        # 解析unittest文件
        unittest_file = _extract_unittest_file(response)
        # unittest_file 可以为 None，因为并非所有响应都包含unittest
            
        return ParsedBugResponse(
            problem_statement=problem_statement,
            bug_analysis=bug_analysis,
            buggy_files=buggy_files,
            unittest_file=unittest_file
        )
        
    except Exception as e:
        print(f"解析错误: {e}")
        return None

def _extract_section(response: str, section_name: str) -> Optional[str]:
    """提取指定节的内容"""
    start_marker = f"==={section_name}_START==="
    end_marker = f"==={section_name}_END==="
    
    start_idx = response.find(start_marker)
    end_idx = response.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        return None
        
    start_idx += len(start_marker)
    content = response[start_idx:end_idx].strip()
    return content

def _extract_buggy_files(response: str) -> List[BugFile]:
    """提取所有buggy文件的信息"""
    buggy_files = []
    
    # 提取BUGGY_FILES部分
    buggy_section = _extract_section(response, "BUGGY_FILES")
    if not buggy_section:
        return buggy_files
    
    # 使用正则表达式找到所有文件块
    file_pattern = r"===FILE_START===(.*?)===FILE_END==="
    file_matches = re.findall(file_pattern, buggy_section, re.DOTALL)
    
    for file_content in file_matches:
        file_info = _parse_single_file(file_content)
        if file_info:
            buggy_files.append(file_info)
            
    return buggy_files

def _parse_single_file(file_content: str) -> Optional[BugFile]:
    """解析单个文件的内容"""
    try:
        # 提取文件路径
        path_match = re.search(r"FILE_PATH:\s*(.+)", file_content)
        if not path_match:
            return None
        file_path = path_match.group(1).strip()
        
        # 提取代码内容
        code_pattern = r"===CODE_START===(.*?)===CODE_END==="
        code_match = re.search(code_pattern, file_content, re.DOTALL)
        if not code_match:
            return None
            
        code_content = code_match.group(1).strip()
        
        # 去除python代码块标记
        if code_content.startswith("```python"):
            code_content = code_content[9:]  # 去除```python
        if code_content.endswith("```"):
            code_content = code_content[:-3]  # 去除```
            
        code_content = code_content.strip()
        
        return BugFile(file_path=file_path, code=code_content)
        
    except Exception:
        return None

def _extract_unittest_file(response: str) -> Optional[UnittestFile]:
    """提取unittest文件的信息"""
    # 提取UNITTEST_FILE部分
    unittest_section = _extract_section(response, "UNITTEST_FILE")
    if not unittest_section:
        return UnittestFile(file_path=None, code=None)
    
    try:
        # 提取文件路径
        path_match = re.search(r"FILE_PATH:\s*(.+)", unittest_section)
        if not path_match:
            return UnittestFile(file_path=None, code=None)
        file_path = path_match.group(1).strip()
        
        # 提取代码内容
        code_pattern = r"===CODE_START===(.*?)===CODE_END==="
        code_match = re.search(code_pattern, unittest_section, re.DOTALL)
        if not code_match:
            return UnittestFile(file_path=None, code=None)
            
        code_content = code_match.group(1).strip()
        
        # 去除python代码块标记
        if code_content.startswith("```python"):
            code_content = code_content[9:]  # 去除```python
        if code_content.endswith("```"):
            code_content = code_content[:-3]  # 去除```
            
        code_content = code_content.strip()
        
        return UnittestFile(file_path=file_path, code=code_content)
        
    except Exception:
        return UnittestFile(file_path=None, code=None)

# ================== Patch Generation Functions ==================

def generate_patches_for_bug_data(data: dict, result: ParsedBugResponse, default_path: str) -> dict:
    """
    根据GPT生成的bug数据，创建临时环境并生成patch
    
    Args:
        data: 原始数据，包含repo、patch等信息
        result: 解析后的GPT响应
        default_path: 默认路径
        
    Returns:
        dict: 包含gpt_patch和test_patch的数据
    """
    # 获取基本信息
    repo_name = data.get('repo', '').replace('/', '__') + '__' + data.get('base_commit', '')[:6]
    original_patch = data.get('patch', '')
    
    # 设置repo路径
    source_testbed = os.path.join(default_path, repo_name)
    
    if not os.path.exists(source_testbed):
        print(f"Source testbed not found: {source_testbed}")
        return data
    
    # 获取GPT生成的bug文件路径
    gpt_bug_files = [bug_file.file_path for bug_file in result.buggy_files]
    
    # 如果有unittest文件，也需要包含在复制列表中
    files_to_copy = gpt_bug_files.copy()
    if result.unittest_file:
        files_to_copy.append(result.unittest_file.file_path)
    
    try:
        with TempTestbed(source_testbed=source_testbed, copy_files=files_to_copy) as temp_testbed:
            temp_dir = temp_testbed.temp_dir
            
            try:
                # 读取修复后的文件内容（正确状态）
                fixed_file_contents = {}
                for bug_file in result.buggy_files:
                    bug_file_path = os.path.join(temp_dir, bug_file.file_path)
                    if os.path.exists(bug_file_path):
                        with open(bug_file_path, 'r', encoding='utf-8') as f:
                            fixed_file_contents[bug_file.file_path] = f.read()
                    else:
                        print(f"Warning: File not found: {bug_file.file_path}")
                        fixed_file_contents[bug_file.file_path] = ""
                
                # 使用fake_git_repo生成从bug到fix的patch
                bug_file_paths = []
                bug_contents = []
                fixed_contents = []
                files_exist_list = []
                
                for bug_file in result.buggy_files:
                    bug_file_paths.append(bug_file.file_path)
                    bug_contents.append(bug_file.code)
                    fixed_contents.append(fixed_file_contents.get(bug_file.file_path, ""))
                    
                    # 检查文件是否存在于临时目录中
                    bug_file_full_path = os.path.join(temp_dir, bug_file.file_path)
                    files_exist_list.append(os.path.exists(bug_file_full_path))
                
                # 生成从bug到fix的patch (bug->fix)
                all_files_exist = all(files_exist_list)
                gpt_patch = fake_git_repo(
                    file_pathes=bug_file_paths,
                    old_contents=bug_contents,
                    new_contents=fixed_contents,
                    files_exist=all_files_exist
                )

                gpt_reverse_patch = fake_git_repo(
                    file_pathes=bug_file_paths,
                    old_contents=fixed_contents,
                    new_contents=bug_contents,
                    files_exist=all_files_exist
                )
                
                # 生成test patch（如果有unittest文件）
                test_patch = ""
                if result.unittest_file:
                    unittest_file_path = os.path.join(temp_dir, result.unittest_file.file_path)
                    unittest_exists = os.path.exists(unittest_file_path)
                    
                    # 读取现有的unittest文件内容（如果存在）
                    existing_unittest_content = ""
                    if unittest_exists:
                        with open(unittest_file_path, 'r', encoding='utf-8') as f:
                            existing_unittest_content = f.read()
                    
                    # 生成test patch（添加或修改unittest文件）
                    test_patch = fake_git_repo(
                        file_pathes=[result.unittest_file.file_path],
                        old_contents=[existing_unittest_content],
                        new_contents=[result.unittest_file.code],
                        files_exist=unittest_exists
                    )
                
                # 更新数据
                data['gpt_patch'] = gpt_patch
                data['gpt_reverse_patch'] = gpt_reverse_patch
                data['test_patch'] = test_patch  # 新增test_patch字段
                data['gpt_problem_statement'] = result.problem_statement
                data['gpt_bug_analysis'] = result.bug_analysis
                
                # 保存unittest文件信息（可选）
                if result.unittest_file:
                    data['unittest_file_path'] = result.unittest_file.file_path
                    data['unittest_file_code'] = result.unittest_file.code
                
            finally:
                # 清理临时patch文件
                pass
                
    except Exception as e:
        print(f"Error processing bug data: {str(e)}")
        
    return data

# ================== Command Execution Functions ==================

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
    process = None
    timed_out = False
    
    try:
        # 创建新进程组  
        process = subprocess.Popen(  
            cmd,  
            shell=True,  
            text=True,  
            preexec_fn=os.setsid  # 创建新会话，使进程成为进程组长  
        )  
        
        # 等待进程完成或超时
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode == 0:  
            return True, False  
        else:  
            logger.error(f"Task {instance_id} failed with return code {process.returncode}")  
            if stderr:
                logger.error(f"stderr: {stderr}")  
            return False, False  
            
    except subprocess.TimeoutExpired:  
        timed_out = True
        logger.error(f"Task {instance_id} timed out")
        
        # 尝试优雅地终止进程
        _terminate_process_group(process, instance_id)
        return False, True
        
    except Exception as e:  
        logger.error(f"Unexpected error in task {instance_id}: {str(e)}")
        
        # 如果进程仍在运行，尝试终止它
        if process and process.poll() is None:
            _terminate_process_group(process, instance_id)
            
        return False, timed_out

def _terminate_process_group(process, instance_id):
    """
    安全地终止进程组
    
    参数:
        process: subprocess.Popen 对象
        instance_id: 实例标识符，用于日志
    """
    if not process or process.poll() is not None:
        # 进程已经结束
        return
        
    try:
        # 获取进程组ID
        pgid = os.getpgid(process.pid)
        
        # 首先尝试SIGTERM信号
        try:
            os.killpg(pgid, signal.SIGTERM)
            logger.debug(f"Sent SIGTERM to process group {pgid} for task {instance_id}")
        except ProcessLookupError:
            # 进程组不存在，可能已经结束了
            logger.debug(f"Process group {pgid} for task {instance_id} already terminated")
            return
        except PermissionError:
            logger.warning(f"Permission denied when terminating process group {pgid} for task {instance_id}")
            return
        
        # 等待进程响应SIGTERM
        try:
            process.wait(timeout=2.0)
            logger.debug(f"Process group {pgid} for task {instance_id} terminated gracefully")
            return
        except subprocess.TimeoutExpired:
            logger.debug(f"Process group {pgid} for task {instance_id} did not respond to SIGTERM, sending SIGKILL")
        
        # 如果SIGTERM无效，使用SIGKILL强制终止
        try:
            os.killpg(pgid, signal.SIGKILL)
            logger.debug(f"Sent SIGKILL to process group {pgid} for task {instance_id}")
        except ProcessLookupError:
            # 进程组已经不存在
            logger.debug(f"Process group {pgid} for task {instance_id} already gone")
            return
        except PermissionError:
            logger.warning(f"Permission denied when force-killing process group {pgid} for task {instance_id}")
            return
        
        # 最后等待进程确实结束
        try:
            process.wait(timeout=1.0)
            logger.debug(f"Process group {pgid} for task {instance_id} force-terminated")
        except subprocess.TimeoutExpired:
            logger.error(f"Failed to terminate process group {pgid} for task {instance_id} even with SIGKILL")
            
    except OSError as e:
        if e.errno == 3:  # No such process
            logger.debug(f"Process {process.pid} for task {instance_id} already terminated")
        else:
            logger.error(f"OS error when terminating process for task {instance_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error when terminating process for task {instance_id}: {e}")

def make_test_command_for_gpt(instance, env_dir, tmp_dir, test_patch, non_test_exts) -> str:
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
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in non_test_exts)
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

# ================== Evaluation Functions ==================

def create_temp_patch_file(patch_content: str, base_path: str, instance_id: str, patch_type: str) -> str:
    """
    创建临时patch文件
    
    Args:
        patch_content: patch内容
        base_path: 基础路径
        instance_id: 实例ID
        patch_type: patch类型（用于文件名）
        
    Returns:
        str: patch文件的完整路径
    """
    patch_path = f"{base_path}/{patch_type}-eval/{instance_id}/{patch_type}_patch.diff"
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    
    with open(patch_path, 'w') as f:
        f.write(patch_content)
    
    return patch_path

def get_repo_commit_name(repo: str, base_commit: str) -> str:
    """获取repo commit名称"""
    return repo.replace("/", "__") + "__" + base_commit[:6]

def merge_files_to_copy(patch_files_list: List[Dict[str, List[str]]]) -> List[str]:
    """
    合并多个patch文件列表
    
    Args:
        patch_files_list: patch文件列表的列表
        
    Returns:
        List[str]: 合并后的文件列表
    """
    all_files = []
    for patch_files in patch_files_list:
        all_files.extend(patch_files.get("modified", []))
        all_files.extend(patch_files.get("added", []))
    
    return list(set(all_files))

def create_error_report(instance_id: str, error_msg: str, timed_out: bool = False, duration: float = 0) -> dict:
    """
    创建错误报告
    
    Args:
        instance_id: 实例ID
        error_msg: 错误消息
        timed_out: 是否超时
        duration: 持续时间
        
    Returns:
        dict: 错误报告
    """
    return {
        instance_id: {
            "success": False,
            "error": error_msg,
            "timed_out": timed_out,
            "duration": duration
        }
    }

def run_parallel_tasks(tasks: List[Dict], log_path: str, exec_func, max_workers: int = 4, timeout: int = 100) -> Dict:
    """
    并行运行多个任务的通用函数
    
    Args:
        tasks: 任务列表
        log_path: 日志路径
        exec_func: 执行函数
        max_workers: 最大工作线程数
        timeout: 超时时间
        
    Returns:
        Dict: 执行结果字典
    """
    results = {}
    complete_tasks = 0
    
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

def save_evaluation_results(results: Dict, results_path: str, eval_type: str):
    """
    保存评估结果
    
    Args:
        results: 结果字典
        results_path: 结果文件路径
        eval_type: 评估类型（用于日志）
    """
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"{eval_type} evaluation results saved to {results_path}")

def retry_unittest_generation_for_task(task: dict, repo_path: str) -> Optional[dict]:
    """
    为单个任务重新生成unittest和buggy code
    
    Args:
        task: 原始任务数据
        repo_path: repo路径
        
    Returns:
        Optional[dict]: 更新后的任务数据，失败时返回None
    """
    instance_id = task.get('instance_id', 'unknown')
    try:
        retry_prompt = construct_unittest_prompt(task)
        
        # 获取新的LLM响应
        new_response = get_model_resposne(retry_prompt)
        if 'API request failed' in new_response:
            logger.warning(f"API request failed for retry of task {instance_id}")
            return None
        
        # 解析新响应
        result = parse_bug_response(new_response)
        if not result:
            logger.warning(f"Failed to parse retry response for task {instance_id}")
            return None
        
        # 生成新的patches
        task_copy = task.copy()
        task_copy['response'] = new_response  # 更新response
        processed_data = generate_patches_for_bug_data(task_copy, result, repo_path)
        
        logger.info(f"Successfully retried unittest generation for task {instance_id}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in retry_unittest_generation for task {task.get('instance_id', 'unknown')}: {e}")
        return None

def retry_tasks_in_parallel(tasks_need_retry: List[dict], repo_path: str) -> List[dict]:
    """
    并行重试多个任务
    
    Args:
        tasks_need_retry: 需要重试的任务列表
        repo_path: repo路径
        
    Returns:
        List[dict]: 重试后的任务列表
    """
    retried_tasks = []
    max_workers = min(CONC, len(tasks_need_retry))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(retry_unittest_generation_for_task, task, repo_path): task
            for task in tasks_need_retry
        }
        
        desc = "Retrying unittest generation"
        with tqdm(total=len(tasks_need_retry), desc=desc, unit="task") as pbar:
            for future in as_completed(future_to_task):
                try:
                    retried_task = future.result()
                    if retried_task:
                        retried_tasks.append(retried_task)
                        logger.info(f"Successfully retried task {retried_task.get('instance_id', 'unknown')}")
                    else:
                        # 如果重试失败，保留原任务
                        original_task = future_to_task[future]
                        retried_tasks.append(original_task)
                        logger.warning(f"Retry failed for task {original_task.get('instance_id', 'unknown')}, keeping original")
                except Exception as e:
                    original_task = future_to_task[future]
                    logger.error(f"Error retrying task {original_task.get('instance_id', 'unknown')}: {e}")
                    retried_tasks.append(original_task)
                finally:
                    pbar.update(1)
    
    return retried_tasks

def retry_buggy_code_generation_for_task(task: dict) -> Optional[dict]:
    """
    Retry generating buggy code for a task using the buggy_retry prompt (single retry only)
    
    Args:
        task: The task to retry
        
    Returns:
        Updated task with new buggy code, or None if failed
    """
    instance_id = task.get('instance_id', 'unknown')
    try:
        # Format the prompt with task-specific information
        formatted_prompt = construct_buggy_prompt(task)
        # print (formatted_prompt)
        # Get LLM response
        response = get_model_resposne(formatted_prompt)
        
        # Parse the response to get new buggy code
        result = parse_bug_response(response)
        if not result:
            logger.warning(f"Failed to parse bug response for {instance_id} during retry")
            return None
        
        # 复用原始的problem statement和unittest，只更新buggy code
        result.problem_statement = task.get('gpt_problem_statement', '')
        result.unittest_file.file_path = task.get('unittest_file_path', '')
        result.unittest_file.code = task.get('unittest_file_code', '')
        
        # Update the task with new buggy code and regenerate patches
        updated_task = generate_patches_for_bug_data(task, result, DEFAULT_PATH)
        
        return updated_task
        
    except Exception as e:
        logger.error(f"Error retrying buggy code generation for {instance_id}: {str(e)}")
        return None

def retry_buggy_code_in_parallel(tasks: List[dict], max_workers: int = CONC) -> List[dict]:
    """Parallel version of buggy code retry (single retry only)"""
    retried_tasks = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(retry_buggy_code_generation_for_task, task): task
            for task in tasks
        }
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Retrying buggy code generation"):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    retried_tasks.append(result)
            except Exception as e:
                instance_id = task.get('instance_id', 'unknown')
                logger.error(f"Task {instance_id} failed during parallel retry: {str(e)}")
    
    return retried_tasks

def process_single_task_with_reconstruction(task: dict, all_perfect_tasks: list, all_tasks: list, repo_path: str) -> dict:
    """
    Attempts to reconstruct the prompt for a task using the new function.
    If successful, uses the reconstructed prompt to get a new response.
    Otherwise, falls back to the original method.
    """
    try:
        # Attempt to reconstruct the prompt
        # logger.info(f"Attempting prompt reconstruction for task {task.get('instance_id', 'unknown')}")
        
        # Use the new reconstruction function
        if len(all_perfect_tasks) < 3:
            all_perfect_tasks = all_tasks
        with open(os.path.join(TRUE_PROJECT_FILE_LOC, task['instance_id'] + ".json")) as f:
            structure = json.load(f)
        reconstructed_prompt = construct_prompt(
            task_item=task,
            repo_path=repo_path,
            sample_data=all_perfect_tasks,
            template_path="three_shot",
            structure = structure
        )
        
        if reconstructed_prompt and reconstructed_prompt.strip():
            # logger.info(f"Successfully reconstructed prompt for task {task.get('instance_id', 'unknown')}")
            
            # Get a new response using the reconstructed prompt
            # print (len(reconstructed_prompt))
            new_response = get_model_resposne(reconstructed_prompt)
            if 'API request failed' in new_response:
                logger.warning(f"API request failed for task {task.get('instance_id', 'unknown')}, using original")
                return task
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

# ================== File I/O Functions ==================

def save_checkpoint_state(checkpoint_data: dict, checkpoint_file: str):
    """保存检查点状态到文件"""
    try:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def load_checkpoint_state(checkpoint_file: str) -> Optional[dict]:
    """从文件加载检查点状态"""
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            logger.info(f"Checkpoint loaded from {checkpoint_file}")
            return checkpoint_data
        else:
            logger.info(f"No checkpoint file found at {checkpoint_file}")
            return None
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None

def save_all_tasks_with_state(all_tasks_data: list, state_file: str):
    """保存所有任务数据（包括已成功和待处理的）"""
    try:
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, 'w', encoding='utf-8') as f:
            for task in all_tasks_data:
                f.write(json.dumps(task, ensure_ascii=False) + '\n')
        logger.info(f"All tasks state saved to {state_file}")
    except Exception as e:
        logger.error(f"Failed to save all tasks state: {e}")

def load_all_tasks_with_state(state_file: str) -> List[dict]:
    """加载所有任务数据"""
    try:
        if os.path.exists(state_file):
            tasks = []
            with open(state_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        tasks.append(json.loads(line))
            logger.info(f"Loaded {len(tasks)} tasks from state file {state_file}")
            return tasks
        else:
            logger.info(f"No state file found at {state_file}")
            return []
    except Exception as e:
        logger.error(f"Failed to load tasks state: {e}")
        return []

def load_processed_ids_from_output(final_output_file: str) -> set:
    """从final_output_file中加载已处理的instance_id列表"""
    processed_ids = set()
    if os.path.exists(final_output_file):
        try:
            with open(final_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed_ids.add(data['instance_id'])
            logger.info(f"Loaded {len(processed_ids)} processed instance IDs from {final_output_file}")
        except Exception as e:
            logger.warning(f"Could not load processed IDs from output file: {e}")
    return processed_ids

def load_perfect_tasks_from_output(final_output_file: str) -> List[dict]:
    """从final_output_file中加载已成功的任务数据"""
    perfect_tasks = []
    if os.path.exists(final_output_file):
        try:
            with open(final_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        perfect_tasks.append(json.loads(line))
            logger.info(f"Loaded {len(perfect_tasks)} perfect tasks from {final_output_file}")
        except Exception as e:
            logger.warning(f"Could not load perfect tasks from output file: {e}")
    return perfect_tasks

def append_results_to_jsonl(perfect_tests_results: dict, original_data: list, output_file: str):
    """
    追加保存结果到jsonl文件（用于增量保存）
    
    Args:
        perfect_tests_results: perfect_tests结果（包含完整评估数据）
        original_data: 原始数据列表
        output_file: 输出文件路径
    """
    # 创建输出数据
    output_data = []
    for data in original_data:
        instance_id = data['instance_id']
        if instance_id in perfect_tests_results:
            result_data = perfect_tests_results[instance_id]
            new_data = data.copy()
            
            # 使用完整的评估结果数据
            init_passed_tests = set(result_data['init_result'].get('tests_status', {}).get('PASSED', []))
            bug_passed_tests = set(result_data['bug_result'].get('tests_status', {}).get('PASSED', []))
            bug_failed_tests = set(result_data['bug_result'].get('tests_status', {}).get('FAILED', []))

            new_data['FAIL_TO_PASS'] = list(init_passed_tests & bug_failed_tests)
            new_data['PASS_TO_PASS'] = list(init_passed_tests & bug_passed_tests)
            output_data.append(new_data)
    
    # 追加保存到jsonl文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 使用追加模式，无缝衔接已有数据
    with open(output_file, 'a', encoding='utf-8') as f:
        for data in output_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    logger.info(f"Appended {len(output_data)} results to {output_file}")
    return output_data

