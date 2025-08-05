#!/usr/bin/env python3

import os
import asyncio
import aiohttp
import json
import csv
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

api_key = os.getenv("GLM_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 GLM_API_KEY")

@dataclass
class APIRequest:
    id: int
    prompt: str
    model: str = "glm-4.5"
    max_tokens: int = 8192
    temperature: float = 0.7

@dataclass
class APIResponse:
    id: int
    prompt: str
    response: str
    success: bool
    error_msg: str = ""
    response_time: float = 0.0
    timestamp: str = ""

class BatchAPIClient:
    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.results: List[APIResponse] = []
        self.completed_count = 0
        self.total_count = 0

    async def call_single_api(self, session: aiohttp.ClientSession, request: APIRequest) -> APIResponse:
        payload = {
            "model": request.model,
            "messages": [
                {"role": "user", "content": request.prompt}
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False
        }

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response_time = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    return APIResponse(
                        id=request.id,
                        prompt=request.prompt,
                        response=content,
                        success=True,
                        response_time=response_time,
                        timestamp=timestamp
                    )
                else:
                    error_text = await response.text()
                    return APIResponse(
                        id=request.id,
                        prompt=request.prompt,
                        response="",
                        success=False,
                        error_msg=f"HTTP {response.status}: {error_text}",
                        response_time=response_time,
                        timestamp=timestamp
                    )
        except Exception as e:
            response_time = time.time() - start_time
            return APIResponse(
                id=request.id,
                prompt=request.prompt,
                response="",
                success=False,
                error_msg=str(e),
                response_time=response_time,
                timestamp=timestamp
            )

    async def batch_call_api(self, requests: List[APIRequest], concurrent_limit: int = 30):
        self.total_count = len(requests)
        self.completed_count = 0
        self.results = []

        print(f"开始批量调用API - 总数: {self.total_count}, 并发数: {concurrent_limit}")

        connector = aiohttp.TCPConnector(
            limit=concurrent_limit * 2,
            limit_per_host=concurrent_limit,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers
        ) as session:
            semaphore = asyncio.Semaphore(concurrent_limit)

            async def bounded_request(request: APIRequest):
                async with semaphore:
                    result = await self.call_single_api(session, request)
                    self.completed_count += 1

                    # 显示进度
                    progress = (self.completed_count / self.total_count) * 100
                    status = "✓" if result.success else "✗"
                    print(f"\r进度: {self.completed_count}/{self.total_count} ({progress:.1f}%) {status} ID:{result.id}", end="", flush=True)

                    return result

            start_time = time.time()
            tasks = [bounded_request(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            # 处理结果
            for result in results:
                if isinstance(result, APIResponse):
                    self.results.append(result)
                elif isinstance(result, Exception):
                    self.results.append(APIResponse(
                        id=-1, prompt="", response="", success=False,
                        error_msg=str(result), timestamp=datetime.now().isoformat()
                    ))

            print(f"\n\n批量调用完成，耗时: {end_time - start_time:.2f} 秒")
            self.print_summary()

    def print_summary(self):
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        total = len(self.results)
        success_count = len(successful)
        fail_count = len(failed)
        success_rate = (success_count / total) * 100 if total > 0 else 0

        if successful:
            avg_response_time = sum(r.response_time for r in successful) / len(successful)
        else:
            avg_response_time = 0

        print(f"\n{'='*50}")
        print(f"批量API调用总结")
        print(f"{'='*50}")
        print(f"总请求数: {total}")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"平均响应时间: {avg_response_time:.2f} 秒")

        if failed:
            print(f"\n失败原因统计:")
            error_counts = {}
            for result in failed:
                error_type = result.error_msg.split(':')[0] if ':' in result.error_msg else result.error_msg
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            for error, count in error_counts.items():
                print(f"  {error}: {count} 次")

def load_dataset(file_path: str) -> List[APIRequest]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    requests = []

    if file_path.suffix.lower() == '.json':
        # JSON格式
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    prompt = item.get('prompt', item.get('question', item.get('text', '')))
                    model = item.get('model', 'glm-4.5')
                    max_tokens = item.get('max_tokens', 1000)
                    temperature = item.get('temperature', 0.7)
                else:
                    prompt = str(item)
                    model = 'glm-4.5'
                    max_tokens = 1000
                    temperature = 0.7

                requests.append(APIRequest(i, prompt, model, max_tokens, temperature))

    elif file_path.suffix.lower() == '.jsonl':
        # JSONL格式
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    prompt = item.get('prompt', item.get('question', item.get('text', '')))
                    model = item.get('model', 'glm-4.5')
                    max_tokens = item.get('max_tokens', 1000)
                    temperature = item.get('temperature', 0.7)
                    requests.append(APIRequest(i, prompt, model, max_tokens, temperature))

    elif file_path.suffix.lower() == '.csv':
        # CSV格式
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                prompt = row.get('prompt', row.get('question', row.get('text', ''))
                model = row.get('model', 'glm-4.5')
                max_tokens = int(row.get('max_tokens', 1000))
                temperature = float(row.get('temperature', 0.7))
                requests.append(APIRequest(i, prompt, model, max_tokens, temperature))

    else:
        # 纯文本格式，每行一个prompt
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    requests.append(APIRequest(i, line))

    return requests

def save_results(results: List[APIResponse], output_path: str):
    output_path = Path(output_path)

    if output_path.suffix.lower() == '.json':
        # JSON格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    elif output_path.suffix.lower() == '.jsonl':
        # JSONL格式
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')

    elif output_path.suffix.lower() == '.csv':
        # CSV格式
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'response', 'success', 'error_msg', 'response_time', 'timestamp'])
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

    print(f"结果已保存到: {output_path}")

async def main():
    print("数据集批量API调用工具")
    print("支持格式: JSON, JSONL, CSV, TXT")
    print()

    # 输入文件路径
    file_path = input("请输入数据集文件路径: ").strip()
    if not file_path:
        print("未输入文件路径")
        return

    try:
        requests = load_dataset(file_path)
        print(f"成功加载 {len(requests)} 个请求")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    if not requests:
        print("数据集为空")
        return

    # 并发设置
    concurrent_limit = input("并发数 (默认30): ").strip()
    concurrent_limit = int(concurrent_limit) if concurrent_limit.isdigit() else 30

    # 开始批量调用
    client = BatchAPIClient(api_key)
    await client.batch_call_api(requests, concurrent_limit)

    # 保存结果
    save_option = input("\n是否保存结果? (y/n): ").strip().lower()
    if save_option in ['y', 'yes', '是']:
        output_path = input("输出文件路径 (默认results.json): ").strip()
        output_path = output_path if output_path else "results.json"
        save_results(client.results, output_path)

if __name__ == "__main__":
    try:
        import aiohttp
    except ImportError:
        print("缺少依赖包: aiohttp")
        print("请运行: pip install aiohttp")
        exit(1)

    asyncio.run(main())