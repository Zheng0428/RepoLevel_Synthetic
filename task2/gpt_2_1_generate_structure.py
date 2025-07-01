import json
from envs import DEFAULT_PATH
import os
import uuid
import ast
import random
import subprocess
from envs import DEFAULT_PATH, NEW_DEFAULT_PATH
from envs import GENERATE_DATA_PATH, OUTPUT_DATA_PATH
import json
import argparse
import tempfile  # 添加tempfile导入

def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()



def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        relative_root = os.path.relpath(root, directory_path)
        # if relative_root == ".":
        #     relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure



def get_project_structure_from_scratch(
    repo_name, commit_id, instance_id, repo_playground
):

    # Generate a temperary folder and add uuid to avoid collision
    # repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    # assert not os.path.exists(repo_playground), f"{repo_playground} already exists"
    # repo_playground = os.path.join(repo_playground, )
    # repo_base_name = repo_name.split("/")[-1]
    structure = create_structure(repo_playground)
    
    # Move everything from structure['.'] to the root level if '.' exists
    if '.' in structure:
        dot_content = structure['.']
        structure.pop('.')  # Remove the '.' key
        structure.update(dot_content)  # Move all content to root level
    
    d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return d

def apply_patches_and_create_buggy_repo(entry, repo_playground, new_repo_playground):
    """
    应用patches并创建包含bug的repo
    
    Args:
        entry: 包含patch信息的数据条目
        repo_playground: 原始repo路径
        new_repo_playground: 新repo路径
        
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        # 检查目标文件夹是否存在，如果存在则删除
        if os.path.exists(new_repo_playground):
            print(f"Target directory {new_repo_playground} already exists, removing it...")
            subprocess.run(f"rm -rf {new_repo_playground}", shell=True, check=True)
        
        # 复制原始repo
        subprocess.run(f"cp -r {repo_playground} {new_repo_playground}", shell=True, check=True)
        
        # 获取patches
        origin_patch = entry.get('origin_patch', '')
        patch = entry.get('patch', '')
        
        if not origin_patch or not patch:
            print(f"Warning: Missing patches for {entry.get('instance_id', 'unknown')}")
            return False
        
        # 第一步：应用origin_patch，让repo变成正确状态
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as origin_patch_file:
            origin_patch_file.write(origin_patch)
            origin_patch_file_path = origin_patch_file.name
        
        # 第二步：创建patch文件用于反向应用
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as patch_file:
            patch_file.write(patch)
            patch_file_path = patch_file.name
        
        try:
            # 应用origin_patch
            origin_patch_cmd = f'cd {new_repo_playground} && git apply --whitespace=nowarn {origin_patch_file_path}'
            result_origin = subprocess.run(origin_patch_cmd, shell=True, capture_output=True, text=True)
            
            if result_origin.returncode != 0:
                print(f"Failed to apply origin_patch for {entry.get('instance_id', 'unknown')}: {result_origin.stderr}")
                return False
                
            # 反向应用patch，获得错误的repo
            reverse_patch_cmd = f'cd {new_repo_playground} && git apply --reverse --whitespace=nowarn {patch_file_path}'
            result_reverse = subprocess.run(reverse_patch_cmd, shell=True, capture_output=True, text=True)
            
            if result_reverse.returncode != 0:
                print(f"Failed to apply reverse patch for {entry.get('instance_id', 'unknown')}: {result_reverse.stderr}")
                return False
                
            print(f"Successfully applied patches for {entry.get('instance_id', 'unknown')}")
            return True
            
        finally:
            # 清理临时patch文件
            os.unlink(origin_patch_file_path)
            os.unlink(patch_file_path)
            
    except Exception as e:
        print(f"Error applying patches for {entry.get('instance_id', 'unknown')}: {str(e)}")
        return False

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Process project structure')
    parser.add_argument('--save-structure', action='store_true', 
                       help='Whether to save project structure to files')
    args = parser.parse_args()
    
    dataset = []
    a = 0
    input_file = f'{GENERATE_DATA_PATH}/8_seg_bug_success_with_noise_gpt4o.jsonl'
    save_structure = f'{GENERATE_DATA_PATH}/structure'
    os.makedirs(NEW_DEFAULT_PATH, exist_ok=True)
    
    # 只有在需要保存structure时才创建目录
    if args.save_structure:
        os.makedirs(save_structure, exist_ok=True)
    
    # output_jsonl_file = f'{GENERATE_DATA_PATH}/10_seg_bug_success_with_noise.jsonl'
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            repo_name = entry['repo']
            commit_id = entry['base_commit']
            instance_id = entry['instance_id']
            
            repo_playground = os.path.join(DEFAULT_PATH, entry['repo_base_name'])
            new_repo_playground = os.path.join(DEFAULT_PATH, entry['instance_id'])
            
            # 使用新的函数来应用patches并创建buggy repo
            success = apply_patches_and_create_buggy_repo(entry, repo_playground, new_repo_playground)
            
            if not success:
                print(f"Skipping {instance_id} due to patch application failure")
                continue
            
            # 根据参数决定是否保存structure
            args.save_structure = True
            if args.save_structure:
                structure = get_project_structure_from_scratch(repo_name, commit_id, instance_id, new_repo_playground)
                with open(f"{save_structure}/{entry['instance_id']}.json", "w") as f_out:
                    json.dump(structure, f_out, indent=4)

