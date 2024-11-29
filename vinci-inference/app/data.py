from typing import Any, Optional
from pydantic import BaseModel, Field

class SeineInferenceRequest(BaseModel):
    prompt: str=Field(title="提示词", description="描述视频如何生成的提示词", example="跳舞")
    image: Optional[Any]=Field(default="",title="图片URL链接或者多维数组", description="", example="https://oss.openmmlab.com/vinci/minimalism-p1.PNG")
    base64_image: Optional[Any]=Field(default="",title="图片base64字符串", description="", example="https://oss.openmmlab.com/vinci/minimalism-p1.PNG")

class SeineInferenceResponse(BaseModel):
    video: Optional[Any]=Field(default="", title="视频URL链接")
    video_base64: Optional[str]=Field(default="")

class InternvlInferenceRequest(BaseModel):
    session_id: Optional[str]=Field("default", title="会话ID", description="同一个会话ID，时间戳必须大于上次的时间戳", example="default")
    timestamp: Optional[int]=Field(0, title="时间戳", description="如果是同一个对话，则时间戳必须大于上次的时间戳", example=0)
    silent: Optional[bool]=Field(False, title="静默模式", description="非应答模式，用于定时截帧时用静默模式，否则使用应答模式", example=False)
    question: str=Field(title="问题", example="视频内容是什么?")
    history: Optional[list]=Field(default=[], title="问答记录", description="每次对话后返回的对话历史记录", example=[])
    frames: Optional[list]=Field(default=[], title="视频帧", description="视频帧的URL链接或者多维数组", example=["https://oss.openmmlab.com/vinci/minimalism-p1.PNG"])
    base64_frames: Optional[list]=Field(default=[], title="视频帧", description="视频帧的base64", example=["https://oss.openmmlab.com/vinci/minimalism-p1.PNG"])

class IntervlInferenceResponse(BaseModel):
    answer: Optional[str]=Field(None, title="回答")
    history: Optional[list]=Field(None, title="问答记录", description="每次对话后返回的对话历史记录")
    session_id: Optional[str]=Field("default", title="会话ID", description="同一个会话ID，时间戳必须大于上次的时间戳", example="default")
