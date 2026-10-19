from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Connection(BaseModel):
    part_a_item: int = Field(description="Item number of first connected part")
    part_b_item: int = Field(description="Item number of second connected part")
    connection_type: str = Field(description="Type of connection e.g. Threaded, Press-fit, Bolted")
    fastener_item: Optional[int] = Field(default=None, description="Item number of fastener used")
    drawing_evidence: str = Field(description="Evidence/view supporting this connection")
    source_snippet: Optional[str] = Field(default=None, description="Exact text snippet, label, or note from the drawing supporting this connection")

class Part(BaseModel):
    item: int = Field(description="Item number from BOM callout")
    name: str = Field(description="Part/Component Name")
    qty: int = Field(description="Quantity required")
    material: Optional[str] = Field(default=None)
    dimensions: Optional[str] = Field(default=None)
    thread_spec: Optional[str] = Field(default=None)
    bounding_box: Optional[List[float]] = Field(default=None, description="[ymin, xmin, ymax, xmax] normalized (0.0 to 1.0)")
    extracted_text_snippet: Optional[str] = Field(default=None, description="Exact raw text read from the BOM row or callout label for strict traceability")
    properties: Dict[str, Any] = Field(default_factory=dict)
    uncertainty_notes: Optional[str] = Field(default=None)

class AssemblyStep(BaseModel):
    step: int
    action: str
    parts: List[int]
    confidence: str
    evidence: str
    source_reference: Optional[str] = Field(default=None, description="Reference note, view number, or exploded indicator supporting this step")

class AssemblyData(BaseModel):
    name: str
    source: str
    material: Optional[str] = None
    scale: Optional[str] = None
    detected_views: List[str] = Field(default_factory=list)
    parts: List[Part] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    assembly_steps: List[AssemblyStep] = Field(default_factory=list)
    unresolved_uncertainties: List[str] = Field(default_factory=list, description="Ambiguous component mappings or unreadable text flagged for review")
    validation_notes: List[str] = Field(default=list, description="Self-validation check notes verifying extracted data against input image")