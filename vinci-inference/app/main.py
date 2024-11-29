import uvicorn
import threading

from fastapi import FastAPI
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

import data
import service

from exception import exception_handlers


router = APIRouter()

@router.post("/inference/internvl", response_model=data.IntervlInferenceResponse)
async def internval_inference(req: data.InternvlInferenceRequest):
   return service.internvl_inference(req.question, req.history, req.base64_frames, req.frames, 
                                     req.session_id, req.timestamp, req.silent)

chat_lock = threading.Lock()
@router.post("/inference/internvl/stream", response_model=data.IntervlInferenceResponse)
async def internval_inference(req: data.InternvlInferenceRequest):
    try:
        chat_lock.acquire()
        stream_generator = service.internvl_stream_inference(req.question, req.history, req.base64_frames, req.frames, 
                                                             req.session_id, req.timestamp, req.silent, model_index=0)
        return EventSourceResponse(stream_generator)
    finally:
        chat_lock.release()

   
screen_shot_chat_lock = threading.Lock()
@router.post("/inference/internvl/screenshot", response_model=data.IntervlInferenceResponse)
async def internval_inference(req: data.InternvlInferenceRequest):
    try:
        chat_lock.acquire()
        return service.internvl_inference(req.question, req.history, req.base64_frames, req.frames, 
                                          req.session_id, req.timestamp, req.silent, model_index=-1)
    finally:
        chat_lock.release()

@router.post("/inference/seine", response_model=data.SeineInferenceResponse)
async def seine_inference(req: data.SeineInferenceRequest):
    return service.seine_inference(req.prompt, req.base64_image, req.image)

def build_app():
    fast_kwargs = {
        "include_in_schema": True,
        "docs_url": '/swagger-ui'}
    app = FastAPI(exception_handlers=exception_handlers, **fast_kwargs)
    app.include_router(router, prefix="/api/v1", tags=["Model"])

    return app

def start_fast_server(*args, **kwarg):
    uvicorn.run(build_app(), host="0.0.0.0", port=int(18081))

if __name__ == "__main__":
    start_fast_server()
