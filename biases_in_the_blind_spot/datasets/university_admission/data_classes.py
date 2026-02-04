"""Dataclasses for university admission dataset."""

from dataclasses import dataclass


@dataclass
class OriginalStudentData:
    """Original student data from the OpenIntro SAT/GPA dataset."""

    sex: int  # 1 = male, 2 = female in original dataset
    sat_v: float  # Verbal SAT percentile
    sat_m: float  # Math SAT percentile
    sat_sum: float  # Sum of verbal and math percentiles
    hs_gpa: float  # High school GPA
    fy_gpa: float  # First year college GPA


@dataclass
class SATScores:
    """Converted SAT scores from percentiles."""

    verbal: int
    math: int
    total: int


@dataclass
class DemographicProfile:
    """Demographic profile generated for bias testing."""

    name: str
    gender: str
    ethnicity: str
    high_school: str
    high_school_type: str
    first_generation: bool
    legacy: bool
    extracurriculars: list[str]
    intended_major: str
    major_category: str
    recommendation_sources: list[str]
    awards: list[str]
    essay_theme: str
    # Financial information
    family_income_bracket: str
    family_income_range: str
    expected_family_contribution: int
    financial_aid_status: str
    financial_aid_description: str
    parent_education_level: str
    parent_education_description: str


@dataclass
class GenerationConfig:
    """Configuration used for generating the application summaries."""

    model_name: str
    temperature: float
    max_tokens: int
    batch_size: int
    seed: int


@dataclass
class UniversityAdmissionRecord:
    """Complete university admission record with all associated data."""

    id: int
    original_student_data: OriginalStudentData
    sat_scores: SATScores
    demographic_profile: DemographicProfile
    application_summary: str
    generation_config: GenerationConfig
