"""Loan approval dataset helpers.

This module provides utilities for loading and working with the loan approval dataset
that includes demographic variations for bias research.

Example usage:
    from biases_in_the_blind_spot.datasets.loan_approval import (
        load_loan_approval_dataset,
        load_approved_loans,
        load_rejected_loans,
        load_loans_by_demographic,
        get_dataset_statistics,
        LoanApprovalRecord,
        DemographicProfile,
        OriginalLoanData,
    )

    # Load all records
    all_loans = load_loan_approval_dataset()

    # Load only approved loans
    approved = load_approved_loans()

    # Load loans by demographic
    white_male_loans = load_loans_by_demographic(ethnicity="white", gender="male")

    # Get dataset statistics
    stats = get_dataset_statistics()
"""

from .data_classes import (
    DemographicProfile,
    GenerationConfig,
    LoanApprovalRecord,
    OriginalLoanData,
)
from .loaders import (
    get_dataset_statistics,
    load_approved_loans,
    load_loan_approval_dataset,
    load_loans_by_demographic,
    load_rejected_loans,
)

__all__ = [
    # Data classes
    "LoanApprovalRecord",
    "OriginalLoanData",
    "DemographicProfile",
    "GenerationConfig",
    # Loader functions
    "load_loan_approval_dataset",
    "load_approved_loans",
    "load_rejected_loans",
    "load_loans_by_demographic",
    "get_dataset_statistics",
]
