import requests
import base64

API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-7B-Instruct"
headers = {"Authorization": f"Bearer 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"}  # replace with your token

# List of your local image paths
image_files = [
    "grill_task_capture/front_rgb.png",
    "grill_task_capture/left_shoulder_rgb.png",
    "grill_task_capture/right_shoulder_rgb.png",
    "grill_task_capture/overhead_rgb.png",
    "grill_task_capture/wrist_rgb.png"
]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "All of these images are from a robot's cameras. Based on these images, describe the scene in detail."}
        ]
    }
]

# Add all local images
for path in image_files:
    with open(path, "rb") as f:
        img_bytes = f.read()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    messages[0]["content"].insert(
        0,  # put before text, order matters
        {"type": "image", "image": base64_image}
    )

payload = {"inputs": messages}

response = requests.post(API_URL, headers=headers, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

try:
    print(response.json())
except requests.exceptions.JSONDecodeError:
    print("Could not decode JSON from response.")
