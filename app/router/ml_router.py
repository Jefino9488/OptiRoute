import re
import structlog
import httpx
from typing import Optional
from pydantic import BaseModel

logger = structlog.get_logger()

class MLRouterPrediction(BaseModel):
    domain: str
    complexity: int
    is_math: bool
    is_code: bool
    route: str  # 'small model' or 'big model'
    justification: str

class MLRouter:
    """Wraps the Supra-Router-51M ML orchestrator model via llama-server HTTP API."""
    
    def __init__(self, server_url: str = "http://localhost:8081/v1"):
        self._server_url = server_url
        self._client = httpx.Client(timeout=10.0)
        self._is_ready = self._check_health()
        
    def _check_health(self) -> bool:
        """Ping llama-server health endpoint."""
        try:
            base = self._server_url.replace("/v1", "")
            resp = self._client.get(f"{base}/health", timeout=2.0)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                return True
        except Exception:
            pass
        logger.warning("ml_router.server_unavailable", url=self._server_url)
        return False
            
    def predict(self, prompt: str) -> Optional[MLRouterPrediction]:
        """Predict the route for a given prompt."""
        if not self._is_ready:
            return None
            
        formatted_prompt = f"Task: {prompt}\nAnalysis: "
        
        try:
            resp = self._client.post(
                f"{self._server_url}/completions",
                json={
                    "prompt": formatted_prompt,
                    "max_tokens": 128,
                    "temperature": 0.0,
                    "stop": ["\n"]
                }
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["text"].strip()
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
