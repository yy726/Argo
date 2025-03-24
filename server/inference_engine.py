import torch

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from pydantic import BaseModel

from model.din import DeepInterestModel
from server.retrieval_engine import RetrievalEngine
from server.feature_store import FeatureStore, FeatureName


# define user request, for now we just use a simple user id
class RecommendationRequest(BaseModel):
    user_id: int

app = FastAPI()

# for now we hard code the model to be used for inference
checkpoint = torch.load('/tmp/din-movie-len-small.pth', weights_only=False)
model = DeepInterestModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# initialize candidate generation and feature store
retrieval_engine = RetrievalEngine(movie_index=checkpoint['movie_index'])
feature_store = FeatureStore(movie_index=checkpoint['movie_index'])

@app.post("/recommend/")
async def recommend(request: RecommendationRequest):
    try:

        # fetch candidates
        candidates = retrieval_engine.generate_candidates()
        user_id = request.user_id

        # fetch features
        user_sequence_feature = feature_store.generate_feature(FeatureName.USER_HISTORY_SEQUENCE_FEATURE,
                                                               user_id)

        # model inference

        # return topk

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))