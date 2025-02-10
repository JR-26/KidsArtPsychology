from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel
# Define the model and adapter paths
model2_name, tokenizer2 = FastLanguageModel.from_pretrained(model_name="unsloth/Phi-3-mini-4k-instruct")
ft_model = PeftModel.from_pretrained(model2_name,r"home\joel\children_drawing_report\models\phi3_child_finetuned").to("cuda")

print("Model and tokenizer loaded successfully.")
