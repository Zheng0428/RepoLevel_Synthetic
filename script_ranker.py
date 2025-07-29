import json
import os
import utils
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def validate_dependencies(ranking_results: dict, valid_files: list) -> dict:
    """验证并清理不存在的依赖项和文件路径"""
    if not ranking_results:
        return ranking_results
    
    cleaned_results = {}
    
    for file_path, file_info in ranking_results.items():
        # 检查文件路径是否存在，不存在则跳过
        if file_path not in valid_files:
            continue
            
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
        # print(f"Processing repository: {repo_name}")

        # 构造排序提示
        prompt, valid_files = utils.script_ranker_prompt(repo_data['structure'], truncate=10000)
        if not prompt:
            print(f"Failed to generate prompt for {repo_name}")
            return None

        # 获取LLM响应
        if len(prompt) > 200000:
            # print(f"Prompt too long for {repo_name}, skipping")
            prompt, valid_files = utils.script_ranker_prompt(repo_data['structure'], truncate=1000)
            if len(prompt) > 200000:
                prompt, valid_files = utils.script_ranker_prompt(repo_data['structure'], truncate=100)
                if len(prompt) > 200000:
                    print(f"Prompt too long for {repo_name}, total {len(valid_files)} files and {len(prompt)} length, skipping")
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


def process_single_record(record: dict, structure_path: str, max_quantity: int) -> dict:
    """处理单个记录，返回处理后的记录"""
    try:
        instance_id = record.get('instance_id')
        if not instance_id:
            return record

        # 查找对应的仓库JSON文件
        json_filename = f"{instance_id}.json"
        json_path = os.path.join(structure_path, json_filename)

        if not os.path.isfile(json_path):
            # 仍将原始记录返回
            return record

        # 处理仓库排序
        ranking_result = process_repo_json(json_path, max_quantity)

        if ranking_result:
            # 添加排序信息到记录
            record['ranker_info'] = ranking_result

        return record
    except Exception as e:
        print(f"Error processing record {record.get('instance_id', 'unknown')}: {str(e)}")
        return record


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

    # 读取现有输出文件中的记录（如果存在）
    existing_records = {}
    if os.path.isfile(args.output_jsonl):
        print(f"Found existing output file: {args.output_jsonl}")
        with open(args.output_jsonl, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                try:
                    record = json.loads(line)
                    instance_id = record.get('instance_id')
                    if instance_id:
                        # 检查是否已有有效的ranker_info
                        if 'ranker_info' in record and record['ranker_info']:
                            existing_records[instance_id] = record
                        else:
                            print(f"Record {instance_id} has no valid ranker_info, will reprocess")
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in existing file line {line_num+1}, skipping")
                    continue

    # 读取输入文件中的所有记录
    all_records = {}
    with open(args.input_jsonl, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile):
            try:
                record = json.loads(line)
                instance_id = record.get('instance_id')
                if instance_id:
                    all_records[instance_id] = record
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in input file line {line_num+1}, skipping")
                continue

    # 确定需要处理的记录
    records_to_process = []
    for instance_id, record in all_records.items():
        if instance_id in existing_records:
            # 使用已有的记录
            records_to_process.append(existing_records[instance_id])
        else:
            # 需要重新处理
            records_to_process.append(record)

    # 过滤出需要重新处理的记录（没有有效ranker_info的）
    need_reprocess = [r for r in records_to_process if not ('ranker_info' in r and r['ranker_info'])]
    
    if need_reprocess:
        print(f"Processing {len(need_reprocess)} records that need reprocessing...")
        
        # 使用多线程处理需要重新处理的记录
        max_workers = min(10, (os.cpu_count() or 1) + 4)
        reprocessed_records = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record = {
                executor.submit(process_single_record, record, args.structure_path, args.max_quantity): record
                for record in need_reprocess
            }
            
            # 处理完成的任务并显示进度条
            with tqdm(total=len(need_reprocess), desc="Processing repositories", unit="repo") as pbar:
                for future in as_completed(future_to_record):
                    try:
                        processed_record = future.result()
                        reprocessed_records.append(processed_record)
                    except Exception as e:
                        original_record = future_to_record[future]
                        print(f"Error processing record {original_record.get('instance_id', 'unknown')}: {str(e)}")
                        reprocessed_records.append(original_record)
                    finally:
                        pbar.update(1)

        # 更新记录列表
        final_records = []
        reprocessed_dict = {r['instance_id']: r for r in reprocessed_records}
        
        for record in records_to_process:
            instance_id = record['instance_id']
            if instance_id in reprocessed_dict:
                final_records.append(reprocessed_dict[instance_id])
            else:
                final_records.append(record)
    else:
        print("All records already have valid ranker_info, skipping processing")
        final_records = records_to_process

    # 写入输出文件
    with open(args.output_jsonl, 'w', encoding='utf-8') as outfile:
        for record in final_records:
            json.dump(record, outfile, ensure_ascii=False)
            outfile.write('\n')

    print(f"Processing complete. Output saved to {args.output_jsonl}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Rank repository scripts by importance using LLM.')

    parser.add_argument('--max_quantity', type=int, default=10,
                        help='Maximum number of scripts to rank (default: 10)')
    parser.add_argument('--structure_path', type=str, default='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data/true_repo_structure',
                        help='Path to directory containing repository JSON files')
    parser.add_argument('--input_jsonl', type=str, default='/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-bench-extra-message-good-20s-8k.jsonl',
                        help='Path to input JSONL file containing instance_id mapping')
    parser.add_argument('--output_jsonl', type=str, default='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data/script_ranker.jsonl',
                        help='Path to output JSONL file with ranking information')
    args = parser.parse_args()
    main(args)

    