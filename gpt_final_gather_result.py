import os  
import json  
import glob  
import argparse
from collections import defaultdict  

# Parse command line arguments
parser = argparse.ArgumentParser(description='Gather and analyze JSONL result files')
parser.add_argument('--result-dir', type=str, required=True, 
                    help='Directory pattern containing JSONL files (e.g., "/path/to/result")')
args = parser.parse_args()

# Directory containing JSONL files  
result_dir = args.result_dir + '/result_*.jsonl'



# Get all JSONL files in the directory  
all_jsonl_files = sorted(glob.glob(result_dir, recursive=True))
total_files = len(all_jsonl_files)  
print(all_jsonl_files[0])

# Values of k to calculate pass@k for  
k_values = [1, 8, 16, 32, 64, 128, total_files]
# Filter k values to only include those that don't exceed the file count  
valid_k_values = [k for k in k_values if k <= total_files]  

print(f"Found {total_files} result files. Will calculate metrics for: {valid_k_values}")  

# Dictionary to store all data for each instance  
instance_data = defaultdict(lambda: {  
    "resolved": [False] * total_files,  
    "non_empty": [False] * total_files  
})  

# Read all files once and populate the data structure  
timed_out_counts = [0] * total_files  
for file_idx, jsonl_file in enumerate(all_jsonl_files):  
    file_name = os.path.basename(jsonl_file)  
    # print(f"Reading file {file_idx+1}/{total_files}: {file_name}")  
    
    with open(jsonl_file, 'r') as f:
        for line in f:  
            try:  
                data = json.loads(line)  
                instance_id = data.get("instance_id")  
                
                # if instance_id in train_ids:
                    # Check if resolved  
                if data.get("result", {}).get("resolved", False):  
                    instance_data[instance_id]["resolved"][file_idx] = True  
                else:
                    instance_data[instance_id]["resolved"][file_idx] = False
                
                # Check if patch is non-empty  
                patch = data.get("patch", "")  
                if patch != "":  
                    instance_data[instance_id]["non_empty"][file_idx] = True  
                
                # Count timed out instances  
                if data.get("result", {}).get("timed_out", False):  
                    timed_out_counts[file_idx] += 1  
                    
            except json.JSONDecodeError:  
                print(f"Error parsing line in {file_name}")  
                continue  

# Calculate and print metrics for each k value  
print("\n" + "="*70)  
print("METRICS FOR DIFFERENT VALUES OF K")  
print("="*70)  

# Final results table  
results = []  

for k in valid_k_values:  
    # Count metrics  
    total_instances = len(instance_data)  
    resolved_instances = sum(1 for inst in instance_data.values() if any(inst["resolved"][:k]))  
    non_empty_instances = sum(1 for inst in instance_data.values() if any(inst["non_empty"][:k]))  
    
    # Calculate k-specific timed out count  
    timed_out_for_k = sum(timed_out_counts[:k])  
    timed_out_rate = timed_out_for_k / (total_instances * k) if total_instances > 0 else 0  
    
    # Calculate intersection (instances resolved in all files they appear in up to k)  
    intersection_count = 0  
    for inst_id, data in instance_data.items():  
        # Check if instance is present in any file up to k  
        present_in_files = False  
        for i in range(k):  
            if data["resolved"][i] or data["non_empty"][i]:  
                present_in_files = True  
                break  
        
        # If instance is present and resolved in all files it appears in up to k  
        if present_in_files:  
            all_resolved = True  
            for i in range(k):  
                # If we find evidence of the instance in a file but it's not resolved there, break  
                if (data["resolved"][i] or data["non_empty"][i]) and not data["resolved"][i]:  
                    all_resolved = False  
                    break  
            
            if all_resolved:  
                intersection_count += 1  
    
    # Store and print results  
    pass_at_k = resolved_instances / total_instances if total_instances > 0 else 0  
    avg_resolved_counts = [  
        sum(inst["resolved"][:k]) / k for inst in instance_data.values()  
    ]  
    avg_at_k = sum(avg_resolved_counts) / total_instances if total_instances > 0 else 0  
    
    print(f"\nRESULTS FOR PASS@{k}:")  
    print(f"Total unique instances: {total_instances}")  
    print(f"Resolved in at least one of first {k} files: {resolved_instances}/{total_instances} ({pass_at_k:.2%})")  
    print(f"Non-empty patches in at least one of first {k} files: {non_empty_instances}/{total_instances} ({non_empty_instances/total_instances:.2%} if total_instances > 0 else 0)")  
    print(f"Timed out rate: {timed_out_rate:.2%}")  
    print(f"Intersection (resolved in all files they appear in): {intersection_count}")  
    print(f"Avg@{k} (平均每实例在前{k}个文件中通过的比例): {avg_at_k:.2%}")  
    
    results.append({  
        "k": k,  
        "total": total_instances,  
        "resolved": resolved_instances,  
        "non_empty": non_empty_instances,  
        "pass_at_k": pass_at_k,  
        "intersection": intersection_count,
        "avg_at_k": avg_at_k
    })  

# Print final comparison table  
print("\n\n" + "="*80)  
print("FINAL COMPARISON OF PASS@K METRICS")  
print("="*80)  
print(f"{'k':<8} {'Total':<10} {'Resolved':<10} {'Non-Empty':<12} {'Non-Empty%':<12} {'Pass@k':<10} {'Avg@k':<10}")  
print("-" * 90)  
for r in results:  
    non_empty_percentage = (r['non_empty'] / r['total'] * 100 if r['total'] != 0 else 0)  # Calculate non-empty percentage
    print(f"{r['k']:<8} {r['total']:<10} {r['resolved']:<10} {r['non_empty']:<12} {non_empty_percentage:.2f}% {r['pass_at_k']:.2%} {r['avg_at_k']:.2%}")
# 保存每个instance id 被解决的次数
for inst_id, data in instance_data.items():
    instance_data[inst_id]["resolved_counts"] = sum(data["resolved"])


from collections import Counter
# 统计每个instance id被解决的次数
resolved_counts = Counter(instance_data[inst_id]["resolved_counts"] for inst_id in instance_data)
# 打印每个instance id被解决的次数, 按照10的倍数分组
for i in range(1, max(resolved_counts.keys())+1, 10):
    print(f"Number of instances resolved {i}-{min(i+9, max(resolved_counts.keys()))} times: {sum(resolved_counts[j] for j in range(i, min(i+10, max(resolved_counts.keys())+1)))}")

print("solve only once: ", resolved_counts[1])
print("solve all: ", resolved_counts[total_files])
print( max(resolved_counts.keys()))

# any_solved_id = []
# for inst_id, data in instance_data.items():
#     if data["resolved_counts"] >0:
#         any_solved_id.append(inst_id)

# with open("any_solved_id.json", "w") as f:
#     json.dump(any_solved_id, f, indent=4)