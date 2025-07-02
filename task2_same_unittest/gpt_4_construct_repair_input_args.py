import json
import random
from transformers import AutoTokenizer
import utils
from prompts import THINKING_SYSTEM, AGENTLESS_REPAIR
from datasets import load_dataset
import argparse

def count_tokens(tokenizer, messages_or_prompt):
    if isinstance(messages_or_prompt, str):
        return len(tokenizer.encode(messages_or_prompt))
    return len(
        tokenizer.apply_chat_template(messages_or_prompt, add_generation_prompt=True)
    )


def get_file_contents_and_tokens(tokenizer, instance_id, origin_instance_id,pred_files: list[str]) -> dict[str, str]:
    structure = utils.get_repo_structure(instance_id)
    repo_file_contents, _, _ = utils.get_full_file_paths_and_classes_and_functions(
        structure
    )
    repo_file_contents_dict = {path: lines for path, lines in repo_file_contents}
    file_contents_and_tokens = {}
    try:
        for pred_file in pred_files:
            content = "\n".join(repo_file_contents_dict[pred_file])
            file_tokens = count_tokens(tokenizer, content)
            file_contents_and_tokens[pred_file] = {
                "content": content,
                "tokens": file_tokens,
            }
    except KeyError as e:
        print(f"Error processing {instance_id}: {e}")
    return file_contents_and_tokens

def get_input_messages(problem_statement, context: str):
    content = AGENTLESS_REPAIR.format(
        problem_statement=problem_statement,
        content=context,
    ).strip()
    messages = [{"role": "system", "content": THINKING_SYSTEM}]
    messages.append({"role": "user", "content": content})
    return messages  


def construct_topn_file_context(
    pred_files: list[str],
    file_contents: dict[str, str],
    max_input_tokens: int,
    # Randomize the order of the contents
    randomize: bool = False,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    num_tokens = 0
    all_contents = list[str]()
    for pred_file in pred_files:
        content = file_contents[pred_file]["content"]
        content = f"### {pred_file}\n{content}"
        num_new_tokens = file_contents[pred_file]["tokens"]
        # if num_tokens + num_new_tokens > max_input_tokens:
        #     continue
        num_tokens += num_new_tokens
        all_contents.append(content)

    # if len(all_contents) == 0 and len(pred_files) > 0:
    #     return f"### {pred_files[0]}\n{file_contents[pred_files[0]]}"
    if randomize:
        random.shuffle(all_contents)
    return "\n\n".join(all_contents), num_tokens



def construct_messages_for_task(task,tokenizer, max_tokens=7000,max_noise_file_num=3):
    instance_id = task["instance_id"]
    origin_instance_id = task["origin_instance_id"]
    problem_statement = task["problem_statement"]
    modified_files = task["modified_files"]
    pred_files = task["extra_related_files"]
    random.shuffle(pred_files)
    file_contents_and_tokens = get_file_contents_and_tokens(tokenizer, instance_id, origin_instance_id, modified_files+pred_files)
    gt_tokens = sum([file_contents_and_tokens[file]["tokens"] for file in modified_files])
    input_files = modified_files
    # if gt_tokens >max_tokens:
    #     print(instance_id, f"does not fit in {max_tokens} tokens, gt_tokens: {gt_tokens}")
    extra_files= []
    if gt_tokens<max_tokens and max_noise_file_num>0:
        for file in pred_files:
            if file_contents_and_tokens[file]["tokens"]+gt_tokens<max_tokens:
                input_files.append(file)
                extra_files.append(file)
                gt_tokens += file_contents_and_tokens[file]["tokens"]
                if  len(extra_files)== max_noise_file_num:
                    break

    context, num_tokens = construct_topn_file_context(
        input_files,
        file_contents_and_tokens,
        max_input_tokens=99999,
        randomize=True,
    )
    messages = get_input_messages(problem_statement, context)
    num_tokens = count_tokens(tokenizer, messages)
    return {"instance_id": instance_id, "input_files": input_files, "noise_files":extra_files, "message_tokens": num_tokens, "messages": messages}
        

def main(tokenizer_path, save_path, dataset_jsonl, max_input_tokens, max_noise_file_num):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print(f"Tokenizer loaded from: {tokenizer_path}")
    print(f"Data will be saved to: {save_path}")
    print(f"Dataset JSONL path: {dataset_jsonl}")
    print(f"Max input tokens set to: {max_input_tokens}")


    dataset = load_dataset("json", data_files=dataset_jsonl)["train"]


    input_dataset = dataset.map(
        lambda x: construct_messages_for_task(x, tokenizer, max_noise_file_num=max_noise_file_num),
        batched=False,
        num_proc=64,
    )
    print("len(input_dataset):", len(input_dataset))


    input_dataset = input_dataset.filter(lambda x: x["message_tokens"]<=max_input_tokens, num_proc=64)
    print(f"filtered out input length > {max_input_tokens}, len(input_dataset):", len(input_dataset))

    input_dataset.to_json(
        save_path,
        orient="records",
        lines=True,
    )
    print(f"Saved to {save_path}")

    # 统计有多少个unique repo
    repo_name = input_dataset["repo"]

    unique_repos = set(repo_name)
    print("unique repos:", len(unique_repos))

    #统计input_files-noise_files的数量分布
    input_files = input_dataset["input_files"]
    input_files_count = [len(files) for files in input_files]
    print("input_files_count min:", min(input_files_count), "max:", max(input_files_count), "mean:", sum(input_files_count)/len(input_files_count))
    #统计noise_files的数量分布
    noise_files = input_dataset["noise_files"]

    noise_files_count = [len(files) for files in noise_files]
    print("noise_files_count min:", min(noise_files_count), "max:", max(noise_files_count), "mean:", sum(noise_files_count)/len(noise_files_count))

    gt_files_count = [input_f - noise for input_f, noise in zip(input_files_count, noise_files_count)]
    print("gt_files_count min:", min(gt_files_count), "max:", max(gt_files_count), "mean:", sum(gt_files_count)/len(gt_files_count))
"""
python /mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/task2/gpt_4_construct_repair_input_args.py \
--tokenizer_path /mnt/hdfs/tiktok_aiic/user/codeai/hf_models/Qwen2.5-Coder-32B-Instruct \
--save_path /mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/prcessed_data_v1/9_ready_train_gpt4o.jsonl \
--dataset_jsonl /mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/prcessed_data_v1/8_seg_bug_success_with_noise_gpt4o.jsonl \
--max_noise_file_num 3
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    # Define the arguments with default values
    parser.add_argument("--tokenizer_path", type=str, 
                        default="/mnt/hdfs/tiktok_aiic/user/codeai/hf_models/Qwen2.5-Math-7B",
                        help="Path to the pretrained tokenizer")
    
    parser.add_argument("--save_path", type=str, 
                        default="",
                        help="Path where the output will be saved")
    
    parser.add_argument("--dataset_jsonl", type=str, 
                        default="/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe_extra_processed/swe-bench-extra-0417-good.jsonl",
                        help="Path to the input dataset in JSONL format")
    
    parser.add_argument("--max_input_tokens", type=int, 
                        default=9000,
                        help="Maximum number of input tokens")
    
    parser.add_argument("--max_noise_file_num", type=int, 
                        default=0,
                        help="Maximum number of extra noise files")

    # Parse the arguments
    args = parser.parse_args()

    if args.save_path == "":
        args.save_path = args.dataset_jsonl.replace(".jsonl", f"-max_in_tokens{args.max_input_tokens}-max_noise_file{args.max_noise_file_num}.jsonl")
    main(args.tokenizer_path, args.save_path, args.dataset_jsonl, args.max_input_tokens, args.max_noise_file_num)
    print ('done')
