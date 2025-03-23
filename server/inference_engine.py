import torch

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from pydantic import BaseModel

from model.din import DeepInterestModel
from server.retrieval_engine import simple_candidate_generation


# define user request, for now we just use a simple user id
class RecommendationRequest(BaseModel):
    user_id: int

app = FastAPI()

# for now we hard code the model to be used for inference
checkpoint = torch.load('/tmp/din-movie-len-small.pth')
model = DeepInterestModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
movie_index = checkpoint['movie_index']

@app.post("/recommend/")
async def recommend(request: RecommendationRequest):
    try:

        # fetch candidates
        candidates = simple_candidate_generation(movie_index)
        user_id = request.user_id

        # fetch features

        # model inference

        # return topk

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))