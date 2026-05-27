from pydantic import BaseModel
from .axiom_rules import AxiomTier


class AxiomChallenge(BaseModel):
    challenger: str
    target: str
    axiom: AxiomTier
    argument: str
    evidence: dict
    requested_action: str


class DebateMessage(BaseModel):
    round_number: int
    sender: str
    message_type: str  # "CHALLENGE", "RESPONSE", "CONCEDE", "MAINTAIN"
    content: str
    axiom_challenge: AxiomChallenge | None = None
