#!/usr/bin/env python3
"""
StarGate Benchmark - v1
TAI Research 定制化模型评测工具
"""

import json
import torch
import argparse
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

class StarGate:
    def __init__(self, model_path, config):
        self.tokenizer = AutoTokenizer.from_pretrained("../model")
        self.config = MiniMindConfig(
            hidden_size=config.get("hidden_size", 1024),
            num_hidden_layers=config.get("num_hidden_layers", 32)
        )
        self.model = MiniMindForCausalLM(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.results = {"passed": [], "failed": []}

    def test(self, case):
        """执行单个测试用例"""
        question = case["user"]
        expected_keywords = case.get("expected_keywords", [])
        forbidden_keywords = case.get("forbidden_keywords", [])
        expected_answer = case.get("expected_answer", None)
        max_new_tokens = case.get("max_new_tokens", 512)

        # 生成回答
        inputs = self.tokenizer(question, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(question):].strip()

        # 判断逻辑
        passed = True
        reasons = []

        # 1. 检查期望关键词
        if expected_keywords:
            for kw in expected_keywords:
                if kw not in response:
                    passed = False
                    reasons.append(f"缺少关键词: {kw}")

        # 2. 检查禁止关键词（防重复）
        if forbidden_keywords:
            for kw in forbidden_keywords:
                if response.count(kw) > 2:
                    passed = False
                    reasons.append(f"重复过多: '{kw}' 出现 {response.count(kw)} 次")

        # 3. 检查期望答案（简单算术或事实）
        if expected_answer:
            if expected_answer not in response:
                passed = False
                reasons.append(f"回答不正确，期望包含: {expected_answer}")

        # 4. 检查重复模式（通用防重复）
        words = response.split()
        if len(words) > 10:
            word_freq = {}
            for w in words:
                if len(w) > 1:
                    word_freq[w] = word_freq.get(w, 0) + 1
            for w, freq in word_freq.items():
                if freq > 3 and len(w) > 2:
                    passed = False
                    reasons.append(f"重复模式: '{w}' 出现 {freq} 次")
                    break

        return {
            "passed": passed,
            "reasons": reasons,
            "response": response[:200] + "..." if len(response) > 200 else response
        }

    def run(self, cases):
        """运行所有测试用例"""
        total = len(cases)
        passed = 0
        failed = 0
        details = []

        for case in cases:
            result = self.test(case)
            if result["passed"]:
                passed += 1
            else:
                failed += 1
            details.append({
                "id": case.get("id", "unknown"),
                "category": case.get("category", "unknown"),
                "question": case["user"][:50] + "...",
                "result": result
            })

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "details": details
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--cases", type=str, default="stargate_cases.jsonl", help="测试用例文件")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    args = parser.parse_args()

    # 加载测试用例
    with open(args.cases, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    config = {"hidden_size": args.hidden_size, "num_hidden_layers": args.num_hidden_layers}
    stargate = StarGate(args.model, config)
    results = stargate.run(cases)

    # 输出报告
    print("="*80)
    print("StarGate Benchmark 评测报告")
    print("="*80)
    print(f"总用例: {results['total']}")
    print(f"通过: {results['passed']}")
    print(f"失败: {results['failed']}")
    print(f"通过率: {results['passed']/results['total']*100:.1f}%")
    print("-"*80)

    if results['failed'] > 0:
        print("\n失败用例详情:")
        for d in results['details']:
            if not d['result']['passed']:
                print(f"  [{d['id']}] {d['question']}")
                print(f"    原因: {', '.join(d['result']['reasons'])}")
                print(f"    回答: {d['result']['response']}")
                print()

    print("="*80)

if __name__ == "__main__":
    main()