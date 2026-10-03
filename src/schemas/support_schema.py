from pydantic import BaseModel
from typing import List


class SupportAnalysis(BaseModel):
    intent: str
    sentiment: str
    urgency: str
    summary: str
    entities: List[str]
    root_cause: str
    recommended_action: str
    department: str
    escalation_required: bool
    customer_response: str