import json
import os
import utils
from pathlib import Path


def process_repo_json(json_path: str, max_quantity: int) -> dict:
    """处理单个仓库JSON文件，生成排序结果"""
    try:
        # 读取仓库结构数据
        with open(json_path, 'r', encoding='utf-8') as f:
            repo_data = json.load(f)
        repo_name = os.path.splitext(os.path.basename(json_path))[0]
        print(f"Processing repository: {repo_name}")

        # 构造排序提示
        prompt = utils.script_ranker_prompt(repo_data['structure'])
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

        return ranking_results
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return None


def extract_ranking(response: str, max_quantity: int) -> list:
    """从LLM响应中提取并验证排序结果"""
    try:
        # 尝试JSON解析
        ranking = json.loads(response)
        if not isinstance(ranking, list):
            raise ValueError("Ranking result must be a list")
        return ranking[:max_quantity]
    except json.JSONDecodeError:
        # 尝试正则表达式提取
        import re
        pattern = r'\d+\.\s*([^\n]+)'
        matches = re.findall(pattern, response)
        return matches[:max_quantity] if matches else None
    except Exception as e:
        print(f"Error extracting ranking: {str(e)}")
        return None


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

    