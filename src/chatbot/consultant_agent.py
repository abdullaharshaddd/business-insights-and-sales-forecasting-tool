import torch
import os
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from src.chatbot.business_toolkit import query_database, get_sales_forecast_summary, get_churn_risk_overview

# 1. Configuration for Local AI (4-bit for 16GB RAM)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
print(f"Initializing Strategic Brain ({model_id})...")

# Check for GPU
device = 0 if torch.cuda.is_available() else -1

# Quantization setup
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config if torch.cuda.is_available() else None,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

# 2. Setup the Reasoning Pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, max_length = None, device=device)

def consult_logic(user_input):
    """A manual Reasoning loop for the Strategic Consultant"""
    
    # System Instruction
    system_prompt = f"""### Instructions
You are a Strategic Business Consultant. You have access to three specific business skills:
1. SQL Search: To find past facts in the Olist Database.
2. Sales Forecast: To see future revenue trends.
3. Churn Risk: To identify at-risk customers.

Think step-by-step about how to use these to help the owner.
User Question: {user_input}
### Response
"""
    
    print("\n[Consultant is thinking deeply about your business...]")
    output = pipe(system_prompt)[0]['generated_text']
    
    # Return everything after the prompt
    return output.split("### Response")[-1].strip()

def start_consultancy():
    print("="*60)
    print("Welcome! Your AI Strategic Consultant is now active.")
    print("Status: Local Reasoning Loop (No LangChain dependency)")
    print("="*60)
    
    while True:
        user_input = input("\n[Owner]: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            response = consult_logic(user_input)
            print(f"\n[AI Consultant]:\n{response}")
            print("\n" + "-"*30)
        except Exception as e:
            print(f"\n[Error]: {str(e)}")

if __name__ == "__main__":
    start_consultancy()
