from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from transformers import AutoTokenizer, TextStreamer, StoppingCriteriaList
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load first model
import os
model1_dir = os.path.abspath("/home/joel/children_drawing_report/models/qwen2vl_2b_instruct_lora_merged_2")
processor1 = AutoProcessor.from_pretrained(model1_dir)
model1 = Qwen2VLForConditionalGeneration.from_pretrained(model1_dir, device_map="auto")

# model2_name, tokenizer2 = FastLanguageModel.from_pretrained(model_name="unsloth/Phi-3-mini-4k-instruct")
# ft_model = PeftModel.from_pretrained(model2_name,r"home\joel\children_drawing_report\models\phi3_child_finetuned").to("cuda")

model2_name, tokenizer2 = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3-mini-4k-instruct",# Ensure quantization is used if applicable
    device_map="auto",
)

# Load the fine-tuned model on the same device
ft_model = PeftModel.from_pretrained(
    model2_name,
    r"home\joel\children_drawing_report\models\phi3_child_finetuned"
).to("cuda")
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    # Save uploaded image
    image_file = request.files["image"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
    image_file.save(image_path)

    # Load and process image for first model
    image = Image.open(image_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image with all the possible details you capture."},
            ],
        }
    ]
    text_prompt = processor1.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor1(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    output_ids = model1.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor1.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    description = output_text[0]

    # Process first model's output through the second model
    prompt = f"""
    You provide psychological inference for the summary of children's drawings.
    According to the summary passed, can you give your subjective psychological views summing up the state of mind and what the child is trying to convey.

    ### Instruction:
    "Give your psychological view ",

    ### Input:
    {description}

    ### Output:
    """
    inputs = tokenizer2([prompt], return_tensors="pt").to("cuda")
    output_ids = model2.generate(**inputs, max_new_tokens=500)
    final_output = tokenizer2.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"description": description, "psychological_inference": final_output})

if __name__ == "__main__":
    app.run(debug=True)
