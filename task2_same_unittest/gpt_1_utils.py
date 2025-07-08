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
from temp_testbed import TempTestbed, get_all_filenames
from utils import fake_git_repo

# Add additional imports for the new function
try:
    from utils.common import read_yaml, count_tokens
except ImportError:
    # Fallback for simpler count_tokens if needed
    def count_tokens(text, model_name="gpt-4o"):
        return len(text.split())
    
    def read_yaml(config='default'):
        import yaml
        if os.path.exists(f'config/prompt/{config}.yaml'):
            yaml_file = f'config/prompt/{config}.yaml'
        else:
            yaml_file = config
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)

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
class ParsedBugResponse:
    """解析后的bug响应结构"""
    problem_statement: str
    bug_analysis: str
    buggy_files: List[BugFile]

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
            
        return ParsedBugResponse(
            problem_statement=problem_statement,
            bug_analysis=bug_analysis,
            buggy_files=buggy_files
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

# ================== Patch Generation Functions ==================

def generate_patches_for_bug_data(data: dict, result: ParsedBugResponse, default_path: str) -> dict:
    """
    根据GPT生成的bug数据，创建临时环境并生成patch
    
    Args:
        data: 原始数据，包含repo、patch等信息
        result: 解析后的GPT响应
        default_path: 默认路径
        
    Returns:
        dict: 包含gpt_patch的数据
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
    
    # 合并需要复制的文件
    files_to_copy = gpt_bug_files
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
                # 如果所有文件都存在，则files_exist=True，否则False
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
                
                # 更新数据
                data['gpt_patch'] = gpt_patch
                data['gpt_reverse_patch'] = gpt_reverse_patch
                data['gpt_problem_statement'] = result.problem_statement
                data['gpt_bug_analysis'] = result.bug_analysis
                
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

def reconstruct_three_shot_prompt(item: dict, default_path: str, all_samples: List[dict], template_path: str = "yimi/three_shot_same_test") -> Optional[str]:
    """
    根据data_loader.py中的逻辑重构three-shot prompt
    
    Args:
        item: 当前需要处理的任务项
        default_path: 仓库根目录路径 (通常是 "/opt/tiger/expr/repo_commit")
        all_samples: 所有可用的样本列表，用于随机选择示例
        template_path: YAML模板路径
        
    Returns:
        重构的prompt字符串，如果失败返回None
    """
    try:
        # 加载模板
        template = read_yaml(template_path)
        
        # 获取基本信息
        example_problem_statement_1 = item['problem_statement']          
        repo_name = item.get('repo').replace('/', '__') + '__' + item.get('base_commit')[:6]
        
        # 获取原始patch来提取示例bug信息
        origin_patch = item['patch']
        
        # 创建临时目录来应用patch并获取修改内容
        source_testbed = os.path.join(default_path, repo_name)
        if not os.path.exists(source_testbed):
            logger.warning(f"Source testbed not found: {source_testbed}")
            return None
        
        # 获取将被patch修改的文件
        patch_files = get_all_filenames(origin_patch)
        modified_files = patch_files["modified"] + patch_files["added"]
        
        if not modified_files:
            logger.warning(f"No modified files found in patch for {item.get('instance_id', 'unknown')}")
            return None
        
        # 使用TempTestbed为第一个示例创建临时环境
        try:
            with TempTestbed(source_testbed=source_testbed, copy_files=modified_files) as temp_testbed:
                temp_dir = temp_testbed.temp_dir
                
                # 将patch写入临时文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as patch_file:
                    patch_file.write(origin_patch)
                    patch_file_path = patch_file.name
                
                try:
                    # 应用patch获取修改后的（有bug的）内容
                    patch_cmd = f'cd {temp_dir} && git apply --whitespace=nowarn {patch_file_path}'
                    result = subprocess.run(patch_cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        logger.warning(f"Failed to apply patch for example 1: {result.stderr}")
                        return None
                    
                    # 读取所有修改的文件来获取示例buggy代码
                    example_buggy_files_1 = ""
                    for file_path in modified_files:
                        example_file_full_path = os.path.join(temp_dir, file_path)
                        
                        if os.path.exists(example_file_full_path):
                            with open(example_file_full_path, 'r') as f:
                                file_content = f.read()
                                example_buggy_files_1 += f"**File Path:** {file_path}\n```python\n{file_content}\n```\n\n"
                        else:
                            continue
                    
                    if not example_buggy_files_1:
                        logger.warning(f"No buggy files content found for example 1")
                        return None
                        
                finally:
                    # 清理临时patch文件
                    os.unlink(patch_file_path)
                    
        except Exception as e:
            logger.error(f"Error processing first example: {e}")
            return None
        
        # 随机采样其他示例，然后选择最短的两个
        available_samples = [s for s in all_samples if s != item]  # 排除当前项
        if len(available_samples) < 3:
            logger.warning(f"Not enough samples for selection: {len(available_samples)}")
            return None
            
        random_samples = random.sample(available_samples, min(5, len(available_samples)))
        
        # 处理其他示例并收集它们的buggy_files长度
        candidate_examples = []
        
        for sample_item in random_samples:
            sample_problem_statement = sample_item['problem_statement']
            sample_repo_name = sample_item.get('repo').replace('/', '__') + '__' + sample_item.get('base_commit')[:6]
            sample_origin_patch = sample_item['patch']
            sample_source_testbed = os.path.join(default_path, sample_repo_name)
            
            if not os.path.exists(sample_source_testbed):
                continue
            
            sample_patch_files = get_all_filenames(sample_origin_patch)
            sample_modified_files = sample_patch_files["modified"] + sample_patch_files["added"]
            
            if not sample_modified_files:
                continue
            
            try:
                with TempTestbed(source_testbed=sample_source_testbed, copy_files=sample_modified_files) as sample_temp_testbed:
                    sample_temp_dir = sample_temp_testbed.temp_dir
                    
                    # 将patch写入临时文件
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as sample_patch_file:
                        sample_patch_file.write(sample_origin_patch)
                        sample_patch_file_path = sample_patch_file.name
                    
                    try:
                        # 应用patch获取修改后的（有bug的）内容
                        sample_patch_cmd = f'cd {sample_temp_dir} && git apply --whitespace=nowarn {sample_patch_file_path}'
                        sample_result = subprocess.run(sample_patch_cmd, shell=True, capture_output=True, text=True)
                        
                        if sample_result.returncode != 0:
                            continue
                        
                        # 读取所有修改的文件获取示例buggy代码
                        sample_example_buggy_files = ""
                        for file_path in sample_modified_files:
                            sample_example_file_full_path = os.path.join(sample_temp_dir, file_path)
                            
                            if os.path.exists(sample_example_file_full_path):
                                with open(sample_example_file_full_path, 'r') as f:
                                    file_content = f.read()
                                    sample_example_buggy_files += f"**File Path:** {file_path}\n```python\n{file_content}\n```\n\n"
                            else:
                                continue
                        
                        if sample_example_buggy_files:
                            candidate_examples.append({
                                'problem_statement': sample_problem_statement,
                                'buggy_files': sample_example_buggy_files,
                                'buggy_files_length': len(sample_example_buggy_files)
                            })
                            
                    finally:
                        # 清理临时patch文件
                        os.unlink(sample_patch_file_path)
                        
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue
        
        # 如果没有足够的候选示例则跳过
        if len(candidate_examples) < 2:
            logger.warning(f"Not enough candidate examples: {len(candidate_examples)}")
            return None
        
        # 按buggy_files长度排序并选择最短的两个
        candidate_examples.sort(key=lambda x: x['buggy_files_length'])
        selected_examples = candidate_examples[:2]
        
        # 获取修补后的文件
        patch_files = get_all_filenames(item['patch'])
        test_files = get_all_filenames(item['test_patch'])
        files_to_copy = list(set(
            patch_files["modified"] + 
            patch_files["added"] +
            test_files["modified"] + 
            test_files["added"]
        ))
        
        patch = item['patch']
        test_patch = item['test_patch']
        
        with TempTestbed(source_testbed=source_testbed, copy_files=files_to_copy) as temp_testbed:
            temp_dir = temp_testbed.temp_dir
            
            # 应用patch
            patch_cmd = f'cd {temp_dir} && git apply --whitespace=nowarn'
            result = subprocess.run(patch_cmd, shell=True, input=patch, text=True, capture_output=True)
            if result.returncode != 0:
                logger.warning(f"Failed to apply main patch: {result.stderr}")
                return None
                
            try:
                test_patch_cmd = f'cd {temp_dir} && git apply --whitespace=nowarn'
                result = subprocess.run(test_patch_cmd, shell=True, input=test_patch, text=True, capture_output=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to apply test patch: {result.stderr}")
                    return None
            except:
                logger.warning("Failed to apply test patch")
                return None

            # 获取原始干净代码（应用patch前）
            original_code = ''
            files_to_modify = []
            test_file = item['input_files'] + item['noise_files']
            for test_file_name in test_file:
                full_file_path = os.path.join(temp_dir, test_file_name)
                if os.path.exists(full_file_path):
                    with open(full_file_path, 'r') as f:
                        file_content = f.read()
                        original_code += f"File Name: {test_file_name}\n\nFile Content:\n ```python\n{file_content}\n```\n"
                        files_to_modify.append(test_file_name)
                else:
                    continue
                    
            if not files_to_modify or not original_code:
                logger.warning("No files to modify or original code found")
                return None

            # 获取unittest代码
            unittest_code = ''
            test_patch_modify_files = test_files["modified"] + test_files["added"] 
            for test_file_name in test_patch_modify_files:
                full_file_path = os.path.join(temp_dir, test_file_name)
                if os.path.exists(full_file_path):
                    with open(full_file_path, 'r') as f:
                        file_content = f.read()
                        unittest_code += f"File Name: {test_file_name}\n\nFile Content:\n ```python\n{file_content}\n```\n"
                else:
                    continue
        
        # 构建最终prompt
        prompt = template['prompt_template'].format(
            original_code=original_code,
            unittest_code=unittest_code,
            example_problem_statement_1=example_problem_statement_1,
            example_buggy_files_1=example_buggy_files_1,
            example_problem_statement_2=selected_examples[0]['problem_statement'],
            example_buggy_files_2=selected_examples[0]['buggy_files'],
            example_problem_statement_3=selected_examples[1]['problem_statement'],
            example_buggy_files_3=selected_examples[1]['buggy_files']
        )

        # 检查token限制
        try:
            if count_tokens(prompt) < 100000:  # 增加三个示例的token限制
                return prompt
            else:
                logger.warning("Prompt exceeds token limit")
                return None
        except:
            logger.warning("Failed to count tokens, using prompt anyway")
            return prompt
            
    except Exception as e:
        logger.error(f"Error reconstructing prompt: {e}")
        return None
