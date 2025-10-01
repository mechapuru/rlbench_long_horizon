import os
import base64
from huggingface_hub import InferenceClient
from pathlib import Path

client = InferenceClient(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    token=os.environ["HF_TOKEN"],
)

# Path to a single local image
image_path = "grill_task_capture/front_rgb.png"

# Prepare the message with one image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image from a robot's camera in detail."}
        ]
    }
]

# Add the single local image
path = Path(image_path)
# Infer mime type from extension
mime_type = f"image/{path.suffix.lstrip('.')}"
with open(path, "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")

messages[0]["content"].append(
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded_image}"
        }
    }
)

try:
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=messages,
        max_tokens=1500,
    )
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")