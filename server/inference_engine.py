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
        user_sequence = feature_store.generate_feature(FeatureName.USER_HISTORY_SEQUENCE_FEATURE,
                                                       user_id)
        user_sequence_feature = user_sequence[:8]  # in model training we use sequence length of 8

        # model inference
        # we use a simple approach to predict all candidates within a single batch, there could be 
        # OOM issue, but in the current movie len small dataset, this should not cause issue on modern laptop
        num_candidates = len(candidates)
        data = {
            "user_id": torch.tensor([[user_id]]).expand(num_candidates, -1),  # B x 1, use expand is memory efficient
            "item_id": torch.tensor(candidates).unsqueeze(1),  # B x 1
            "user_history_behavior": torch.tensor(user_sequence_feature).expand(num_candidates, -1),  # B x seq_len
            "user_history_length": torch.tensor([8]).expand(num_candidates, -1),  # B x 1
            "dense_features": torch.ones([8]).expand(num_candidates, -1),  # B x num_dense
        }
        with torch.no_grad():
            prediction = model(data)  # B x 1
        
        sorted_scores, sorted_indices = torch.sort(prediction.squeeze(1), descending=True)
        top_k_scores = sorted_scores[:10].tolist()
        top_k_indices = sorted_indices[:10].tolist()

        # return topk
        return {"recommendations": [
            {
                "movie_id": idx,
                "scores": score,
                "rank": i
            } for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores))
        ]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)