from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model_id = "/home/puruojha/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    
     # Good practice to include this
)

print("Model loaded.")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)
print("Processor loaded.")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# List of your local image paths
image_files = [
    # "grill_task_capture/front_rgb.png",
    "grill_task_capture/left_shoulder_rgb.png",
    "grill_task_capture/right_shoulder_rgb.png",
    # "grill_task_capture/overhead_rgb.png",
    # "grill_task_capture/wrist_rgb.png"
]

# Construct the messages payload
messages = [
    {
        "role": "user",
        "content": []
    }
]

# Add all image paths to the content
for image_path in image_files:
    messages[0]["content"].append({"type": "image", "image": image_path})

# Add the text prompt after the images
messages[0]["content"].append({"type": "text", "text": "Describe the Scene in detail, How many objects are there, what are they, what is the spatial relationship between them, and what is the robot doing?. Where is the Plate located,"})


# Preparation for inference
print("Preparing for inference...")
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
print("Inference...")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("--- Response ---")
print(output_text[0])