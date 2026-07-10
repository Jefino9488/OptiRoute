import re
import structlog
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from llama_cpp import Llama

logger = structlog.get_logger()

class MLRouterPrediction(BaseModel):
    domain: str
    complexity: int
    is_math: bool
    is_code: bool
    route: str  # 'small model' or 'big model'
    justification: str

class MLRouter:
    """Wraps the Supra-Router-51M ML orchestrator model for fast local inference."""
    
    def __init__(self, model_path: Path):
        self._model_path = model_path
        self._llm: Optional[Llama] = None
        self._is_ready = False
        
        self._initialize()
        
    def _initialize(self) -> None:
        """Load the model synchronously into memory. Takes ~50ms for 51M model."""
        if not self._model_path.exists():
            logger.warning("ml_router.missing_model", path=str(self._model_path))
            return
            
        try:
            # We initialize without heavy GPU offload since it's only 51M parameters
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=4096,
                verbose=False,
            )
            self._is_ready = True
            logger.info("ml_router.initialized", path=str(self._model_path))
        except Exception as e:
            logger.error("ml_router.init_failed", error=str(e))
            
    def predict(self, prompt: str) -> Optional[MLRouterPrediction]:
        """Predict the route for a given prompt.
        
        Returns None if the router is not ready, or if the output is malformed.
        """
        if not self._is_ready or not self._llm:
            return None
            
        formatted_prompt = f"Task: {prompt}\nAnalysis: "
        
        try:
            output = self._llm(
                formatted_prompt,
                max_tokens=128,
                temperature=0.0,
                stop=["\n"],
            )
            response_text = output["choices"][0]["text"].strip()
            logger.debug("ml_router.raw_output", output=response_text)
            
            return self._parse_output(response_text)
        except Exception as e:
            logger.error("ml_router.inference_failed", error=str(e))
            return None
            
    def _parse_output(self, text: str) -> Optional[MLRouterPrediction]:
        """Parse the deterministic pipe-separated string from the model.
        
        Expected format:
        Domain: [Semantic Field] | Complexity: [1-5] | Math: [True/False] | Code: [True/False] | Route: [small model/big model] | Justification: [Rule-driven infrastructure reasoning]
        """
        try:
            parts = [p.strip() for p in text.split("|")]
            
            # Simple key-value extraction
            data = {}
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    data[k.strip().lower()] = v.strip()
                    
            if "route" not in data:
                return None
                
            return MLRouterPrediction(
                domain=data.get("domain", "Unknown"),
                complexity=int(data.get("complexity", "3")),
                is_math=data.get("math", "False").lower() == "true",
                is_code=data.get("code", "False").lower() == "true",
                route=data["route"].lower(),
                justification=data.get("justification", ""),
            )
        except Exception as e:
            logger.warning("ml_router.parse_failed", text=text, error=str(e))
            return None
