"""University admission dataset helpers.

This module provides utilities for loading and working with the university admission dataset
that includes demographic and financial variations for bias research.

Example usage:
    from biases_in_the_blind_spot.datasets.university_admission import (
        load_university_admission_dataset,
        load_first_generation_students,
        load_legacy_students,
        load_students_by_demographic,
        load_students_by_financial_status,
        load_students_by_academics,
        get_dataset_statistics,
        UniversityAdmissionRecord,
        DemographicProfile,
        OriginalStudentData,
        SATScores,
    )

    # Load all records
    all_students = load_university_admission_dataset()

    # Load only first-generation students
    first_gen = load_first_generation_students()

    # Load students by demographic
    asian_students = load_students_by_demographic(ethnicity="asian")

    # Load students by financial status
    low_income = load_students_by_financial_status(income_bracket="low")

    # Load students by academics
    high_achievers = load_students_by_academics(min_gpa=3.5, min_sat=1400)

    # Get dataset statistics
    stats = get_dataset_statistics()
"""

from .data_classes import (
    DemographicProfile,
    GenerationConfig,
    OriginalStudentData,
    SATScores,
    UniversityAdmissionRecord,
)
from .generate_university_admission_dataset import UniversityAdmissionDatasetGenerator
from .loaders import (
    get_dataset_statistics,
    load_first_generation_students,
    load_legacy_students,
    load_students_by_academics,
    load_students_by_demographic,
    load_students_by_financial_status,
    load_university_admission_dataset,
)

__all__ = [
    # Data classes
    "UniversityAdmissionRecord",
    "OriginalStudentData",
    "SATScores",
    "DemographicProfile",
    "GenerationConfig",
    # Generator
    "UniversityAdmissionDatasetGenerator",
    # Loader functions
    "load_university_admission_dataset",
    "load_first_generation_students",
    "load_legacy_students",
    "load_students_by_demographic",
    "load_students_by_financial_status",
    "load_students_by_academics",
    "get_dataset_statistics",
]
