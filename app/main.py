import logging
import os

from fastapi import FastAPI

from app.schemas import RefineRequest, RefineResponse, SearchRequest, SearchResponse
from app.services.refiner import QueryRefinerAgent
from app.services.search_pipeline import SearchPipeline
from app.services.embedder import warmup as warmup_embedder
from app.services.vector_store import VectorStoreClient

# Suppress noisy third-party logs
for _logger_name in ("httpx", "sentence_transformers", "transformers", "huggingface_hub", "torch"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Log to both console and file
logger = logging.getLogger("blueknight")
logger.setLevel(logging.DEBUG)

# Console: minimal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))

# File: detailed with timestamps
file_handler = logging.FileHandler("pipeline.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Prevent duplicate logs from root logger
logging.basicConfig(level=logging.WARNING)

app = FastAPI(title="Refiner and Reranker")


@app.on_event("startup")
async def startup() -> None:
    """Pre-load the FAISS index and embedding model so first request is fast."""
    VectorStoreClient.get_instance()
    warmup_embedder()
    logger.info("Startup complete — FAISS index and embedding model loaded")


@app.post("/agent/refine", response_model=RefineResponse)
async def refine(request: RefineRequest) -> RefineResponse:
    """Iterative refinement: refine -> search -> evaluate -> loop or return."""
    agent = QueryRefinerAgent()
    return await agent.refine(request)


@app.post("/search/run", response_model=SearchResponse)
async def search_run(request: SearchRequest) -> SearchResponse:
    """Three-stage pipeline: vector recall -> post-filter -> re-rank."""
    pipeline = SearchPipeline()
    return await pipeline.run(request)
