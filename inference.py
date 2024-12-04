import json
import requests
from datasets import load_dataset
from typing import List
from multiprocessing import Pool

EOF_STRINGS = ["\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>"]

def truncate_after_eof_strings(text: str) -> str:
    """Truncate generated text at any EOF string"""
    for eof in EOF_STRINGS:
        if eof in text:
            text = text[:text.index(eof)]
    return text

def generate_for_prompt(args) -> dict:
    """为单个prompt生成样本"""
    task_id, prompt = args
    generations = []
    
    for i in range(1):  # n_samples = 1
        response = requests.post(
            "http://127.0.0.1:9000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "prompt": prompt,
                "temperature": 0.2,
                "top_p": 0.95,
                "max_tokens": 2048
            }
        )
        
        if response.status_code == 200:
            generation = response.json()['choices'][0]['text']
            clean_code = truncate_after_eof_strings(generation)
            generations.append(clean_code)
        else:
            print(f"Error for task {task_id}, sample {i}: {response.status_code}")
    
    return {
        "task_id": task_id,
        "prompt": prompt,
        "output": generations
    }

def main():
    difficulties = ['ALL']
    taco = load_dataset('BAAI/TACO', split='test', trust_remote_code=True)
    
    prompts = []
    for sample in taco:
        prompt = "\nQUESTION:\n"
        prompt += sample["question"]
        starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"]
        try:
            input_output = json.loads(sample["input_output"])
            fn_name = None if not input_output.get("fn_name") else input_output["fn_name"]
        except ValueError:
            fn_name = None
            
        if starter_code:
            prompt += starter_code
        if (not fn_name) and (not starter_code):
            prompt += "\nUse Standard Input format"
        else:
            prompt += "\nUse Call-Based format"
        prompt += "\nANSWER:\n"
        prompts.append(prompt)

    print(f"Processing {len(prompts)} problems...")
    
    tasks = list(enumerate(prompts))
    
    with Pool(512) as pool:
        results = pool.map(generate_for_prompt, tasks)
    
    results.sort(key=lambda x: x["task_id"])
    
    with open("generation.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main() 