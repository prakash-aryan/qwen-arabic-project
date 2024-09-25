import requests
import json
import time

def generate_response(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # Set to False to get the full response at once
    }
    
    start_time = time.time()
    response = requests.post(url, json=data, stream=True)
    end_time = time.time()
    
    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    response_data = json.loads(decoded_line)
                    if 'response' in response_data:
                        full_response += response_data['response']
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {decoded_line}")
        return full_response.strip(), end_time - start_time
    else:
        return f"Error: {response.status_code}", 0

def compare_models(base_model, fine_tuned_model, prompts):
    print(f"Comparing {base_model} and {fine_tuned_model}\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        
        base_response, base_time = generate_response(base_model, prompt)
        print(f"\n{base_model} Response:")
        print(base_response)
        print(f"Generation time: {base_time:.2f} seconds")
        
        ft_response, ft_time = generate_response(fine_tuned_model, prompt)
        print(f"\n{fine_tuned_model} Response:")
        print(ft_response)
        print(f"Generation time: {ft_time:.2f} seconds")
        
        print("\n" + "-"*50 + "\n")

# Define the model names
base_model = "qwen2:1.5b"  # Original Qwen model
fine_tuned_model = "qwen-arabic-custom"  # Your fine-tuned model

# Define test prompts (in Arabic)
test_prompts = [
    "ما هي عاصمة المملكة العربية السعودية؟",
    "اشرح لي مفهوم الجاذبية بطريقة بسيطة.",
    "ما هي فوائد ممارسة الرياضة بانتظام؟",
    "كيف يمكنني تحسين مهاراتي في اللغة العربية؟",
    "ما هي أهم الأحداث التاريخية في العالم العربي خلال القرن العشرين؟"
]

# Run the comparison
compare_models(base_model, fine_tuned_model, test_prompts)