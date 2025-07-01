import subprocess
import threading
import time
import signal
import os
from grading_simple import get_eval_report, get_eval_report_synthetic
import sys
import json
import re
from datasets import load_dataset
from pathlib import Path
import logging
from temp_testbed import TempTestbed, get_all_filenames

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union, Tuple

GENERATE_DATA_PATH = "/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/prcessed_data_v1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Constants - Miscellaneous

date = time.strftime("%Y-%m-%d-%H")

EXP_PATH=f"/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/test_log/py-patch-test_commit_{date}_origin"
# EXP_PATH=f"/opt/tiger/Github-Repo/test_log/py-patch-test_commit_{date}_2"
# assert not os.path.exists(EXP_PATH), "Experiment path already exists"
    
EXPR_PATH = os.getenv("EXPR_PATH", "/opt/tiger/expr")
ENV_DIR=f"{EXPR_PATH}/conda_env/"
REPO_DIR=f"{EXPR_PATH}/repo_commit/"

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

origin_data_path = "/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-bench-extra-message-good-20s-8k.jsonl"
origin_task = load_dataset("json", data_files=origin_data_path, split="train")
all_tasks = {}  # 使用字典而不是列表
for data in origin_task:
    repo = f'{data["repo"].replace("/","__")}__{data["environment_setup_commit"][:6]}'
    all_tasks[repo] = data  # 使用repo作为键，data作为值


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
            logger.error(f"stderr: {stderr.decode('utf-8', errors='replace')}")  
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



def make_test_command(instance,env_dir,tmp_dir) -> str:
    pytest_path = os.path.join(env_dir, "bin", "pytest")
    test_cmd = (
        # f"{pytest_path} --no-header -rA --tb=no "
        f"{pytest_path} --no-header -rA "
        f"-p no:cacheprovider "
        # f"-n {os.cpu_count()} "  # 使用CPU核心数
        # f"--dist=loadfile "      # 文件级别分发
        # f"--max-worker-restart=0 "  # 禁止重启
        f"--basetemp={tmp_dir} "
        f"-W ignore::DeprecationWarning"
    )
    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = instance["test_patch"]
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




def eval_instance(instance: dict, log_path: str, timeout=100) -> dict:
    """
    Evaluate a single instance
    
    Args:
        instance: Dictionary containing instance information
        log_path: Base path for logging
        is_init: Whether this is an initialization run
        timeout: Maximum execution time in seconds
        
    Returns:
        dict: Evaluation report
    """
    repository = instance['repository']
    eval_patch = instance.get("eval_patch", None)  # 使用正确的键名
    test_patch = instance.get("test_patch", None)  # 使用正确的键名
    instance = all_tasks[repository]
    instance['test_patch'] = test_patch
    instance_id = instance['instance_id']
    repo = instance['repo']
    base_commit = instance['base_commit']
    repo_commit = repository
    


    test_patch_path = f"/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/eval_task_commit/swe-bench-extra-synthetic/{instance_id}/test_patch.diff"
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(test_patch_path), exist_ok=True)
    
    # Write test patch content to file
    if test_patch:
        with open(test_patch_path, 'w') as f:
            f.write(test_patch)
    if eval_patch:
        with open(eval_patch) as f:
            patch = f.read()
        modified_files = get_all_filenames(patch)["modified"]
    else:
        modified_files = []
    test_patch_files = get_all_filenames(test_patch)
    test_patch_files = test_patch_files["modified"]+test_patch_files['added']+test_patch_files["removed"]
    # import pdb;pdb.set_trace()
    source_testbed = os.path.join(REPO_DIR,repo_commit)
    conda_path = os.path.join(ENV_DIR,repo_commit)
        # 设置评估脚本路径
    eval_sh = "/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/eval_task_commit/eval.sh"

    # 检查脚本是否存在
    if not os.path.exists(eval_sh):
        raise FileNotFoundError(f"Evaluation script not found: {eval_sh}")

    with TempTestbed(source_testbed=source_testbed,copy_files=modified_files+test_patch_files) as temp_testbed:
        temp_dir = temp_testbed.temp_dir # 临时repo路径
        # test_name = test_patch_files[0]
        # test_file_path = os.path.join(temp_dir, test_name)
        # with open(test_file_path, 'w') as f:
        #     pass
        temp_pytest = temp_testbed.temp_pytest
        test_command = make_test_command(instance,conda_path,temp_pytest)
        # 确保使用绝对路径
        log_path = os.path.abspath(log_path)
        os.makedirs(log_path, exist_ok=True)
        instance_log = os.path.join(log_path, f"{instance_id}.log")

        cmd = f'bash {eval_sh} {temp_dir} "{test_command}" {test_patch_path}'
        if eval_patch:
            cmd += f' "{eval_patch}" '
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
            # logger.error(f"Error evaluating instance {instance_id}: {str(e)}")
            report =  {
                instance_id: {
                    "success": False,
                    "error": str(e),
                    "timed_out": False
                }
            }
        end_time = time.time()
        duration = end_time - start_time
        report[instance_id]["duration"] = duration

    return report


                    
def eval_parallel_tasks(tasks: List[Dict], log_path, max_workers: int = 4, timeout=100) -> List[Dict]:
    """Run multiple tasks in parallel"""
    results = {}
    complete_tasks=0
    exec_func = eval_instance
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(exec_func, task, log_path, timeout): task
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            result = future.result()
            instance_id = list(result.keys())[0]
            result_value = result[instance_id]
            
            # Merge test status if instance_id already exists
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
            instance_id = list(result.keys())[0]
            result_value = result[instance_id]
            status = "SUCCESS" if result_value["success"] else "FAILED"
            logger.info(
                f"Progress: {complete_tasks}/{len(tasks)} -"
                f"Task {instance_id} {status} - "
                f"Duration: {result_value['duration']:.2f}s"
            )

    
    return results
    
def test_init(tasks,max_workers,timeout):
    import time
    log_path = f"{EXP_PATH}/init_log"
    os.makedirs(log_path, exist_ok=True)


    # cmd = f'bash {eval_sh} {instance_id} {base_commit} "{test_command}" > {test_output} 2>&1'
    start_time = time.time()
    results = eval_parallel_tasks(tasks, log_path, max_workers=max_workers, timeout=timeout)

    total_time = time.time() - start_time

    logger.info("\nExecution Summary:")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average task time: {total_time/len(results):.2f}s")
    bad_tasks = {}
    good_tasks = {}
    for key, value in results.items():
        if not value["success"] or not value.get('tests_status', None):
            try:
                print(key, value)
            except BlockingIOError:
                import time
                time.sleep(0.1)  # 短暂等待后重试
                print(key, value)  # 重试
            bad_tasks.update({key: value})
        elif len(value["tests_status"]["PASSED"])!=0:
            good_tasks[key] = value['tests_status']['PASSED']
        else:
            bad_tasks[key] = value['tests_status']['FAILED']
    print(f"good tasks: {len(good_tasks)}, wrong tasks: {len(bad_tasks)}")

    bad_task_path = f"{EXP_PATH}/init_bad_tasks.json"
    if os.path.exists(bad_task_path):
        with open(bad_task_path, "r") as f:
            bad_tasks.update(json.load(f))
    with open(bad_task_path, "w") as f:
        json.dump(bad_tasks, f, indent=4)
    
    good_tasks_path = f"{EXP_PATH}/init_good_tasks.json"
    if os.path.exists(good_tasks_path):
        with open(good_tasks_path, "r") as f:
            good_tasks.update(json.load(f))

    with open(good_tasks_path, "w") as f:
        json.dump(good_tasks, f, indent=4)
    



# 使用示例：
if __name__ == "__main__":
    start_time = time.time()

    
    # data_path = "/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-bench-extra-message-good-20s-8k-wo-noisefile.jsonl"
    data_path = f'{GENERATE_DATA_PATH}/3_unitest_yimi_testpatch.jsonl'

    

    tasks = load_dataset("json", data_files=data_path, split="train")
    
    # Filter tasks based on conditions
    filtered_tasks = []
    for task in tasks:
        # Check if repository exists in all_tasks
        if task['repository'] not in all_tasks:
            logger.warning(f"Skipping task {task.get('instance_id', 'unknown')}: repository not found in all_tasks")
            continue
            
        # Get test patch files
        test_patch = task.get("test_patch", None)
        if not test_patch:
            logger.warning(f"Skipping task {task.get('instance_id', 'unknown')}: no test patch found")
            continue
            
        # test_patch_files = get_all_filenames(test_patch)
        # test_patch_files = test_patch_files["modified"] + test_patch_files["removed"]
        
        # # Check if there are any test files
        # if not test_patch_files:
        #     logger.warning(f"Skipping task {task.get('instance_id', 'unknown')}: no test files found in patch")
        #     continue
            
        filtered_tasks.append(task)
    
    logger.info(f"Filtered {len(tasks) - len(filtered_tasks)} tasks, {len(filtered_tasks)} tasks remaining")
    
    # Use filtered_tasks instead of original tasks
    test_init(filtered_tasks, 32, 45)
    # test_gold(filtered_tasks, 32, 45)
    print(f"result saved to {EXP_PATH}")


    


    