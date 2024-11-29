import json
import requests
import sseclient

from typing import List

def chat(question: str, frames: List[str], session_id: str, silent=False):
    post_body = {
        "session_id": session_id,
        "timestamp": 12,
        "silent": silent,
        "question": question,
        "history": [],
        "frames": []
    }

    url = "http://10.140.0.243:18081/api/v1/inference/internvl/stream"
    body_json = json.dumps(post_body, ensure_ascii=False)
    print(f"chat sse url: {url}, body: {body_json}")

    with requests.post(url, data=body_json.encode(), stream=True) as response:
        if response.status_code != 200:
            raise Exception("intern chat sse failed", response)
        client = sseclient.SSEClient(response)
        for event in client.events():
            if len(event.data) != 0 and event.data != "Connection closed":
                yield json.loads(event.data)

if __name__ == "__main__":
    frames = ["https://oss.openmmlab.com/vinci/minimalism-p1.PNG"] * 3
    for data in chat(question="视频内容是什么?", frames=frames, session_id="1", silent=False):
        print(data)