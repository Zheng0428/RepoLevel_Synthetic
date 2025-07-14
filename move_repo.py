import pandas as pd
import shutil
import os

def copy_repositories(parquet_file_path, src_base_path, dest_base_path):
    """
    读取 Parquet 文件，并根据其中的 'instance_id' 复制文件夹。

    参数:
    parquet_file_path (str): Parquet 文件的完整路径。
    src_base_path (str): 源文件夹的基础路径。
    dest_base_path (str): 目标文件夹的基础路径。
    """
    print(f"正在读取 Parquet 文件: {parquet_file_path}")

    try:
        # 使用 pandas 读取 Parquet 文件
        df = pd.read_parquet(parquet_file_path)
    except FileNotFoundError:
        print(f"错误：Parquet 文件未找到，请检查路径：{parquet_file_path}")
        return
    except Exception as e:
        print(f"读取 Parquet 文件时发生错误: {e}")
        return

    # 确保目标基础路径存在，如果不存在则创建
    if not os.path.exists(dest_base_path):
        print(f"目标基础路径不存在，正在创建: {dest_base_path}")
        os.makedirs(dest_base_path)

    total_records = len(df)
    print(f"共找到 {total_records} 条记录。")

    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        try:
            # 假设 'data' 列是字典或可以被正确解析
            instance_id = row['instance_id']

            # 构建源和目标路径
            src_dir = os.path.join(src_base_path, instance_id)
            dest_dir = os.path.join(dest_base_path, instance_id)

            print(f"[{index + 1}/{total_records}] 正在处理 instance_id: {instance_id}")

            # 检查源文件夹是否存在
            if not os.path.exists(src_dir):
                print(f"  -> 警告：源文件夹不存在，跳过: {src_dir}")
                continue

            # 检查目标文件夹是否已存在，避免重复复制
            if os.path.exists(dest_dir):
                print(f"  -> 警告：目标文件夹已存在，跳过: {dest_dir}")
                continue

            # 复制文件夹
            print(f"  -> 正在复制: {src_dir} \n     到: {dest_dir}")
            shutil.copytree(src_dir, dest_dir)
            print(f"  -> 复制完成。")

        except KeyError:
            print(f"  -> 错误：在第 {index + 1} 行中未找到 'instance_id' 键。")
        except Exception as e:
            print(f"  -> 处理 instance_id '{instance_id}' 时发生未知错误: {e}")

    print("\n所有操作已完成。")

if __name__ == '__main__':
    # ----- 请在这里配置您的路径 -----
    PARQUET_FILE = '/mnt/hdfs/tiktok_aiic/user/tianyu/rl_datasets/swe-verified/verl_data/test.parquet'
    SOURCE_BASE_DIR = '/opt/tiger/expr/repo_commit'
    DESTINATION_BASE_DIR = '/opt/tiger/expr/synthetic_repo_commit'
    # ---------------------------------

    copy_repositories(PARQUET_FILE, SOURCE_BASE_DIR, DESTINATION_BASE_DIR)