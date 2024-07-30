import os
import requests
import base64
from PIL import Image
from io import BytesIO

# 获取OpenAI API密钥
api_key = os.environ["OPENAI_API_KEY"]
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

metaprompt = '''
- For any marks mentioned in your answer, please highlight them with [].
'''    

# 编码PIL图像为Base64字符串
def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 准备API请求的有效载荷
def prepare_inputs(message, image):
    base64_image = encode_image_from_pil(image)

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": metaprompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 800
    }

    return payload

# 发送请求并返回响应
def request_gpt4v(message, image):
    payload = prepare_inputs(message, image)
    response = requests.post("https://tbnx.plus7.plus/v1/chat/completions", headers=headers, json=payload)
    
    # 打印响应内容以调试
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")
    
    # 检查响应是否成功
    if response.status_code != 200:
        print(f"Request failed: {response.status_code}")
        print(response.text)
        response.raise_for_status()

    try:
        response_json = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Failed to decode JSON from response")
        raise

    # 检查响应内容
    if 'choices' not in response_json:
        print(f"Unexpected response format: {response_json}")
        raise KeyError("Key 'choices' not found in the response")
    
    res = response_json['choices'][0]['message']['content']
    return res
