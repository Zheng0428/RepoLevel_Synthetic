import json
import os
import utils
from pathlib import Path


def validate_dependencies(ranking_results: dict, valid_files: list) -> dict:
    """验证并清理不存在的依赖项"""
    if not ranking_results:
        return ranking_results
    
    cleaned_results = {}
    
    for file_path, file_info in ranking_results.items():
        # 创建新的文件信息副本
        cleaned_info = file_info.copy()
        
        # 获取原始依赖列表
        original_deps = file_info.get('dependencies', [])
        
        # 过滤掉不存在的依赖项
        valid_deps = [dep for dep in original_deps if dep in valid_files]
        
        # 如果有不存在的依赖项被移除，打印日志
        removed_deps = set(original_deps) - set(valid_deps)
        # if removed_deps:
        #     print(f"Removed non-existent dependencies for {file_path}: {removed_deps}")
        
        # 更新依赖列表
        cleaned_info['dependencies'] = valid_deps
        cleaned_results[file_path] = cleaned_info
    
    return cleaned_results


def process_repo_json(json_path: str, max_quantity: int) -> dict:
    """处理单个仓库JSON文件，生成排序结果"""
    try:
        # 读取仓库结构数据
        with open(json_path, 'r', encoding='utf-8') as f:
            repo_data = json.load(f)
        repo_name = os.path.splitext(os.path.basename(json_path))[0]
        print(f"Processing repository: {repo_name}")

        # 构造排序提示
        prompt, valid_files = utils.script_ranker_prompt(repo_data['structure'])
        if not prompt:
            print(f"Failed to generate prompt for {repo_name}")
            return None

        # 获取LLM响应
        len_prompt = len(prompt)
        if len_prompt > 100000:
            print(f"Prompt too long for {repo_name}, skipping")
            return None
        response = utils.get_llm_response(prompt, temperature=0.7)
        if not response:
            print(f"No response from LLM for {repo_name}")
            return None

        # 提取排序结果
        ranking_results = extract_ranking(response, max_quantity)
        if not ranking_results:
            print(f"Failed to extract ranking for {repo_name}")
            return None

        # 验证依赖项
        ranking_results = validate_dependencies(ranking_results, valid_files)

        return ranking_results
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return None


def extract_ranking(response: str, max_quantity: int) -> dict:
    """从LLM响应中提取并验证排序结果，返回字典格式"""
    try:
        import re
        
        # 提取===RANKING_START===和===RANKING_END===之间的内容
        ranking_pattern = r'===RANKING_START===(.*?)===RANKING_END==='
        ranking_match = re.search(ranking_pattern, response, re.DOTALL)
        if not ranking_match:
            print("No ranking section found in response")
            return {}
            
        ranking_content = ranking_match.group(1)
        
        # 提取所有FILE_START块
        file_pattern = r'===FILE_START===(.*?)===FILE_END==='
        file_matches = re.findall(file_pattern, ranking_content, re.DOTALL)
        
        if not file_matches:
            print("No file entries found in ranking")
            return {}
            
        results = {}
        for file_content in file_matches[:max_quantity]:
            content = file_content.strip()
            
            # 提取FILE_PATH
            path_pattern = r'FILE_PATH:\s*(.+?)(?=\n|$)'
            path_match = re.search(path_pattern, content)
            
            # 提取DEPENDENCIES
            deps_pattern = r'DEPENDENCIES:\s*(.+?)(?=\n[A-Z]|$)'
            deps_match = re.search(deps_pattern, content, re.DOTALL)
            
            # 提取IMPORTANCE_SCORE
            score_pattern = r'IMPORTANCE_SCORE:\s*([0-9.]+)'
            score_match = re.search(score_pattern, content)
            
            # 提取REASONING
            reasoning_pattern = r'REASONING:\s*(.+?)(?=\n[A-Z]|$)'
            reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
            
            if path_match:
                file_path = path_match.group(1).strip()
                
                # 解析依赖列表
                dependencies = []
                if deps_match:
                    deps_str = deps_match.group(1).strip()
                    if deps_str and deps_str.lower() not in ['none', '[]', '']:
                        # 处理列表格式，移除方括号和引号
                        deps_str = deps_str.strip('[]')
                        dependencies = [dep.strip().strip('"\'') for dep in deps_str.split(',') if dep.strip()]
                
                importance_score = 0.0
                if score_match:
                    try:
                        importance_score = float(score_match.group(1))
                    except ValueError:
                        importance_score = 0.0
                
                reasoning = ""
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                
                # 构建字典结构
                results[file_path] = {
                    'importance_score': importance_score,
                    'reasoning': reasoning,
                    'dependencies': dependencies
                }
        
        return results
        
    except Exception as e:
        print(f"Error extracting ranking: {str(e)}")
        return {}


def main(args):
    # 验证输入目录和文件
    if not os.path.isdir(args.structure_path):
        print(f"Error: Invalid directory - {args.structure_path}")
        return

    if not os.path.isfile(args.input_jsonl):
        print(f"Error: Input file not found - {args.input_jsonl}")
        return

    # 创建输出目录
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 处理输入JSONL并生成输出
    with open(args.input_jsonl, 'r', encoding='utf-8') as infile, \
         open(args.output_jsonl, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile):
            try:
                # 解析输入JSONL记录
                record = json.loads(line)
                instance_id = record.get('instance_id')
                if not instance_id:
                    print(f"Warning: Missing instance_id in line {line_num+1}, skipping")
                    continue

                # 查找对应的仓库JSON文件
                json_filename = f"{instance_id}.json"
                json_path = os.path.join(args.structure_path, json_filename)

                if not os.path.isfile(json_path):
                    print(f"Warning: JSON file not found for {instance_id}, skipping")
                    # 仍将原始记录写入输出
                    json.dump(record, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    continue

                # 处理仓库排序
                ranking_result = process_repo_json(json_path, args.max_quantity)

                if ranking_result:
                    # 添加排序信息到记录
                    record['ranker_info'] = ranking_result

                # 写入输出文件
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write('\n')

            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in line {line_num+1}, skipping")
                continue
            except Exception as e:
                print(f"Error processing line {line_num+1}: {str(e)}")
                continue

    print(f"Processing complete. Output saved to {args.output_jsonl}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Rank repository scripts by importance using LLM.')

    parser.add_argument('--max_quantity', type=int, default=5,
                        help='Maximum number of scripts to rank (default: 5)')
    parser.add_argument('--structure_path', type=str, default='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data/true_repo_structure',
                        help='Path to directory containing repository JSON files')
    parser.add_argument('--input_jsonl', type=str, default='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data/gpt-4o-2024-11-20_yimi_three_shot_same_test.jsonl',
                        help='Path to input JSONL file containing instance_id mapping')
    parser.add_argument('--output_jsonl', type=str, default='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data/script_ranker.jsonl',
                        help='Path to output JSONL file with ranking information')
    args = parser.parse_args()
    main(args)

    