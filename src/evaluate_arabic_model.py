import requests
import json
import time

def generate_response(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
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

def evaluate_model(model_name, prompts):
    print(f"Evaluating model: {model_name}\n")
    
    for i, (topic, prompt) in enumerate(prompts.items(), 1):
        print(f"Topic {i}: {topic}")
        print(f"Prompt: {prompt}")
        
        response, generation_time = generate_response(model_name, prompt)
        print(f"\nResponse:")
        print(response)
        print(f"Generation time: {generation_time:.2f} seconds")
        
        print("\n" + "-"*50 + "\n")

# Define your model name
model_name = "qwen-arabic-merged-full:latest"  # Replace with your model's name in Ollama

# Define the prompts
arabic_prompts = {
    "التاريخ والحضارة": "اشرح تأثير الحضارة الإسلامية على العلوم والفنون في العصور الوسطى.",
    "الأدب العربي": "قارن بين أسلوب نجيب محفوظ وغسان كنفاني في الكتابة الروائية.",
    "اللهجات العربية": "ما هي الاختلافات الرئيسية بين اللهجة المصرية واللهجة الخليجية؟ أعط أمثلة.",
    "الفلسفة الإسلامية": "ناقش أفكار ابن رشد حول العلاقة بين الدين والفلسفة.",
    "الاقتصاد في العالم العربي": "كيف أثرت أسعار النفط على التنمية الاقتصادية في دول الخليج العربي؟",
    "الفن الإسلامي": "اشرح أهمية الخط العربي في الفن الإسلامي وتطوره عبر العصور.",
    "القضايا الاجتماعية": "ما هي التحديات التي تواجه المرأة العربية في سوق العمل اليوم؟",
    "الطب في التراث العربي": "تحدث عن إسهامات ابن سينا في مجال الطب وتأثيرها على الطب الحديث.",
    "البيئة والتغير المناخي": "كيف تؤثر ظاهرة الاحتباس الحراري على المناطق الصحراوية في العالم العربي؟",
    "التكنولوجيا والابتكار": "ما هي أبرز المبادرات التكنولوجية في العالم العربي وكيف تساهم في التنمية الاقتصادية؟",
    "الشعر العربي المعاصر": "قارن بين أسلوب محمود درويش ونزار قباني في كتابة الشعر.",
    "السياسة والعلاقات الدولية": "كيف تطورت العلاقات بين الدول العربية والاتحاد الأوروبي في العقد الأخير؟"
}

# Run the evaluation
if __name__ == "__main__":
    evaluate_model(model_name, arabic_prompts)
