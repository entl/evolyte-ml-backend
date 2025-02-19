from pydantic import BaseModel

# Health check response
class HealthResponse(BaseModel):
    status: str