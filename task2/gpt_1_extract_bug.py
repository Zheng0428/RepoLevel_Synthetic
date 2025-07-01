import json
import re
import os # Import os for file existence checks
import json5, simplejson
import subprocess
from utils import fake_git_repo
from envs import DEFAULT_PATH
from envs import GENERATE_DATA_PATH, OUTPUT_DATA_PATH
from typing import Dict, List, Optional
from dataclasses import dataclass
from temp_testbed import TempTestbed, get_all_filenames
import tempfile
# The clean_text function is defined but not used in the main logic provided.
# Keep it if it's used elsewhere or remove if completely unused.
def clean_text(text):
    # Remove special characters from the beginning and end, including spaces, newline characters, asterisks, quotes, and colons.
    return re.sub(r'^[\s\*\n\'\"""''：:]+|[\s\*\n\'\"""''：:]+$', '', text)

@dataclass
class BugFile:
    """表示一个包含bug的文件"""
    file_path: str
    code: str

@dataclass
class ParsedBugResponse:
    """解析后的bug响应结构"""
    problem_statement: str
    buggy_files: List[BugFile]
    unittest_file_path: str
    unittest_code: str

def parse_bug_response(response: str) -> Optional[ParsedBugResponse]:
    """
    解析符合 prompt_generate_bug_v1.yaml 格式的响应
    
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
            
        # 解析buggy文件
        buggy_files = _extract_buggy_files(response)
        if not buggy_files:
            return None
            
        # 解析unittest
        unittest_info = _extract_unittest(response)
        if not unittest_info:
            return None
            
        return ParsedBugResponse(
            problem_statement=problem_statement,
            buggy_files=buggy_files,
            unittest_file_path=unittest_info[0],
            unittest_code=unittest_info[1]
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

def _extract_unittest(response: str) -> Optional[tuple[str, str]]:
    """提取unittest信息，返回(file_path, code)"""
    unittest_section = _extract_section(response, "UNITTEST")
    if not unittest_section:
        return None
        
    try:
        # 提取文件路径
        path_match = re.search(r"FILE_PATH:\s*(.+)", unittest_section)
        if not path_match:
            return None
        file_path = path_match.group(1).strip()
        
        # 提取代码内容
        code_pattern = r"===CODE_START===(.*?)===CODE_END==="
        code_match = re.search(code_pattern, unittest_section, re.DOTALL)
        if not code_match:
            return None
            
        code_content = code_match.group(1).strip()
        
        # 去除python代码块标记
        if code_content.startswith("```python"):
            code_content = code_content[9:]
        if code_content.endswith("```"):
            code_content = code_content[:-3]
            
        code_content = code_content.strip()
        
        return (file_path, code_content)
        
    except Exception:
        return None

def _resolve_unittest_file_conflict(unittest_file_path: str, temp_dir: str) -> str:
    """
    解决unittest文件路径冲突问题
    
    Args:
        unittest_file_path: 原始的unittest文件路径
        temp_dir: 临时目录路径
        
    Returns:
        str: 解决冲突后的文件路径
    """
    # 检查是否与temp_dir中已存在的文件冲突
    full_path = os.path.join(temp_dir, unittest_file_path)
    if os.path.exists(full_path):
        # 如果冲突，在文件名后添加'_'
        if unittest_file_path.endswith('.py'):
            new_path = unittest_file_path[:-3] + '_.py'
        else:
            new_path = unittest_file_path + '_'
        return new_path
    
    return unittest_file_path

def generate_patches_for_bug_data(data: dict, result: ParsedBugResponse) -> dict:
    """
    根据GPT生成的bug数据，创建临时环境并生成patch
    
    Args:
        data: 原始数据，包含repo、patch等信息
        result: 解析后的GPT响应
        
    Returns:
        dict: 包含gpt_patch和gpt_test_patch的数据
    """
    # 获取基本信息
    repo_name = data.get('repo', '').replace('/', '__') + '__' + data.get('base_commit', '')[:6]
    original_patch = data.get('patch', '')
    
    # 设置repo路径
    default_path = "/opt/tiger/expr/repo_commit"
    source_testbed = os.path.join(default_path, repo_name)
    
    if not os.path.exists(source_testbed):
        print(f"Source testbed not found: {source_testbed}")
        return data
    
    # 获取原始patch修改的文件
    original_patch_files = get_all_filenames(original_patch)
    modified_files = original_patch_files["modified"] + original_patch_files["added"]
    
    # 获取GPT生成的bug文件路径
    gpt_bug_files = [bug_file.file_path for bug_file in result.buggy_files]
    
    # 合并需要复制的文件
    files_to_copy = list(set(modified_files + gpt_bug_files))
    
    try:
        with TempTestbed(source_testbed=source_testbed, copy_files=files_to_copy) as temp_testbed:
            temp_dir = temp_testbed.temp_dir
            
            # 解决unittest文件路径冲突
            resolved_unittest_path = _resolve_unittest_file_conflict(
                result.unittest_file_path, 
                temp_dir
            )
            
            # 如果路径被修改了，打印提示信息
            if resolved_unittest_path != result.unittest_file_path:
                print(f"Unittest file path conflict resolved: {result.unittest_file_path} -> {resolved_unittest_path}")
            
            # 第一步：应用原始patch，让repo变成正确状态
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as original_patch_file:
                original_patch_file.write(original_patch)
                original_patch_file_path = original_patch_file.name
            
            try:
                # 应用原始patch
                patch_cmd = f'cd {temp_dir} && git apply --whitespace=nowarn {original_patch_file_path}'
                result_apply = subprocess.run(patch_cmd, shell=True, capture_output=True, text=True)
                
                if result_apply.returncode != 0:
                    print(f"Failed to apply original patch: {result_apply.stderr}")
                    return data
                
                # 第二步：读取修复后的文件内容（正确状态）
                fixed_file_contents = {}
                for bug_file in result.buggy_files:
                    bug_file_path = os.path.join(temp_dir, bug_file.file_path)
                    if os.path.exists(bug_file_path):
                        with open(bug_file_path, 'r', encoding='utf-8') as f:
                            fixed_file_contents[bug_file.file_path] = f.read()
                    else:
                        print(f"Warning: File not found after applying original patch: {bug_file.file_path}")
                        fixed_file_contents[bug_file.file_path] = ""
                
                # 第三步：使用fake_git_repo生成从bug到fix的patch
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
                
                # 第四步：生成unittest patch (新建文件)，使用解决冲突后的路径
                gpt_test_patch = fake_git_repo(
                    file_pathes=[resolved_unittest_path],
                    old_contents=[""],
                    new_contents=[result.unittest_code],
                    files_exist=False
                )
                
                # 更新数据
                data['gpt_patch'] = gpt_patch
                data['gpt_reverse_patch'] = gpt_reverse_patch
                data['gpt_test_patch'] = gpt_test_patch
                data['gpt_problem_statement'] = result.problem_statement
                
            finally:
                # 清理临时patch文件
                os.unlink(original_patch_file_path)
                
    except Exception as e:
        print(f"Error processing bug data: {str(e)}")
        
    return data

# --- Modified process_jsonl function ---
def process_jsonl(input_path, output_path):
    default_path = "/opt/tiger/expr/repo_commit"
    a=b=c=d=e_=f_=0
    false_n = 0           # Counter for responses with bad/missing JSON bugs
    total_repo = 0
    total_script = 0
    total_unit = 0
    total_mini_bug = 0
    total_lines_processed = 0
    total_valid_responses = 0
    num_processed_lines_in_file = 0
    # --- Loop through each input file path provided ---
    if not os.path.exists(input_path):
        print(f"Warning: Input file not found: {input_path}. Skipping.")
        input_path = input_path.replace(".tmp", "") # Remove the.tmp extension if it exists
    aggregated_data = []  # List to store aggregated results
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        # --- Process each line in the current file ---
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                num_processed_lines_in_file += 1

                # Extract key fields for processing and aggregation
                meta_response = json.loads(data.get('meta_response'))
                response = meta_response['choices'][0]['message']['content']
                
                # Skip if response is not a string (e.g., already a dict or None)
                # Or if it's an empty string (can't contain bugs)
                if not isinstance(response, str) or not response:
                    continue
                    
                print(f"Processing line {line_num}: {data.get('instance_id', 'unknown')}")
                print('#'*50)
                
                result = parse_bug_response(response)
                print (result.buggy_files[0].code)
                if result is not None:
                    # 处理生成patch的逻辑
                    processed_data = generate_patches_for_bug_data(data, result)
                    aggregated_data.append(processed_data)
                    print(f"Successfully processed and generated patches for {data.get('instance_id', 'unknown')}")
                else:
                    print(f"Failed to parse bug response for {data.get('instance_id', 'unknown')}")
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}")
                continue
            
        print(f"Finished processing {num_processed_lines_in_file} lines from {input_path}.")

    print(f"Processing {input_path}")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Write each repository's complete aggregated data as one JSON line
        for repo_data in aggregated_data:
            # Remove the original prompt and response to save space
            repo_data.pop('prompt', None)
            repo_data.pop('api_resposne', None)
            repo_data.pop('meta_response', None)
            outfile.write(json.dumps(repo_data, ensure_ascii=False) + '\n')
    print(f"\n处理完成，聚合后的数据已保存到 {output_path}")

    # --- Save the final aggregated data to the output file ---
    print(f"\n--- Aggregation Summary ---")
    print(f"Total Repos: {len(aggregated_data)}")

# --- Main execution block ---
if __name__ == "__main__":
    # --- Define Input and Output Paths ---
    # MODIFIED: Define a LIST of input files
    input_jsonl_files = f'{OUTPUT_DATA_PATH}/gpt-4o-2024-11-20_yimi_prompt_generate_bug_v1.jsonl.tmp'
    # Output path remains a single file
    output_jsonl_file = f'{GENERATE_DATA_PATH}/6_bug_gpt4o.jsonl' # Changed name slightly for clarity

    process_jsonl(input_jsonl_files, output_jsonl_file)
    