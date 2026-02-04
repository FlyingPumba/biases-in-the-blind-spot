"""Dataclasses for loan approval dataset."""

from dataclasses import dataclass


@dataclass
class OriginalLoanData:
    """Original loan data from the Kaggle dataset."""

    text: str
    income: int
    credit_score: int
    loan_amount: int
    dti_ratio: float
    employment_status: str
    approval: str


@dataclass
class DemographicProfile:
    """Demographic profile generated for bias testing."""

    name: str
    age: str
    gender: str
    ethnicity: str
    location: str
    location_type: str
    occupation: str
    occupation_type: str


@dataclass
class GenerationConfig:
    """Configuration used for generating the enhanced descriptions."""

    model_name: str
    temperature: float
    max_tokens: int
    batch_size: int
    seed: int


@dataclass
class LoanApprovalRecord:
    """Complete loan approval record with all associated data."""

    id: int
    original_loan_data: OriginalLoanData
    demographic_profile: DemographicProfile
    original_text: str
    enhanced_description: str
    approval_status: str
    generation_config: GenerationConfig
