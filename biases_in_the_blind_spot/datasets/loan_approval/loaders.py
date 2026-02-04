"""Helper functions to load loan approval dataset."""

import json
from pathlib import Path

from biases_in_the_blind_spot.util import PROJECT_ROOT_PATH, ensure_decompressed

from .data_classes import (
    DemographicProfile,
    GenerationConfig,
    LoanApprovalRecord,
    OriginalLoanData,
)


def load_loan_approval_dataset(
    file_path: Path | str | None = None,
) -> list[LoanApprovalRecord]:
    """Load the loan approval dataset and return a list of LoanApprovalRecord dataclasses.

    Args:
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of LoanApprovalRecord objects.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if file_path is None:
        file_path = (
            PROJECT_ROOT_PATH
            / "biases_in_the_blind_spot/datasets/data/loan_approval_dataset.json"
        )

    file_path = Path(file_path)
    file_path = ensure_decompressed(file_path)

    with open(file_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for item in raw_data:
        # Parse original loan data
        original_data = item["original_loan_data"]
        original_loan_data = OriginalLoanData(
            text=original_data["Text"],
            income=original_data["Income"],
            credit_score=original_data["Credit_Score"],
            loan_amount=original_data["Loan_Amount"],
            dti_ratio=original_data["DTI_Ratio"],
            employment_status=original_data["Employment_Status"],
            approval=original_data["Approval"],
        )

        # Parse demographic profile
        demo_data = item["demographic_profile"]
        demographic_profile = DemographicProfile(
            name=demo_data["name"],
            age=demo_data["age"],
            gender=demo_data["gender"],
            ethnicity=demo_data["ethnicity"],
            location=demo_data["location"],
            location_type=demo_data["location_type"],
            occupation=demo_data["occupation"],
            occupation_type=demo_data["occupation_type"],
        )

        # Parse generation config
        config_data = item["generation_config"]
        generation_config = GenerationConfig(
            model_name=config_data["model_name"],
            temperature=config_data["temperature"],
            max_tokens=config_data["max_tokens"],
            batch_size=config_data["batch_size"],
            seed=config_data["seed"],
        )

        # Create the complete record
        record = LoanApprovalRecord(
            id=item["id"],
            original_loan_data=original_loan_data,
            demographic_profile=demographic_profile,
            original_text=item["original_text"],
            enhanced_description=item["enhanced_description"],
            approval_status=item["approval_status"],
            generation_config=generation_config,
        )

        records.append(record)

    return records


def load_approved_loans(
    file_path: Path | str | None = None,
) -> list[LoanApprovalRecord]:
    """Load only the approved loan records.

    Args:
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of LoanApprovalRecord objects with approval_status == "Approved".
    """
    all_records = load_loan_approval_dataset(file_path)
    return [record for record in all_records if record.approval_status == "Approved"]


def load_rejected_loans(
    file_path: Path | str | None = None,
) -> list[LoanApprovalRecord]:
    """Load only the rejected loan records.

    Args:
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of LoanApprovalRecord objects with approval_status == "Rejected".
    """
    all_records = load_loan_approval_dataset(file_path)
    return [record for record in all_records if record.approval_status == "Rejected"]


def load_loans_by_demographic(
    ethnicity: str | None = None,
    gender: str | None = None,
    location_type: str | None = None,
    occupation_type: str | None = None,
    file_path: Path | str | None = None,
) -> list[LoanApprovalRecord]:
    """Load loan records filtered by demographic criteria.

    Args:
        ethnicity: Filter by ethnicity (e.g., "white", "black", "hispanic", "asian", "middle_eastern").
        gender: Filter by gender ("male" or "female").
        location_type: Filter by location type ("affluent", "middle_class", "working_class").
        occupation_type: Filter by occupation type ("professional", "middle_class", "working_class").
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of LoanApprovalRecord objects matching the demographic criteria.
    """
    all_records = load_loan_approval_dataset(file_path)

    filtered_records = []
    for record in all_records:
        profile = record.demographic_profile

        # Apply filters
        if ethnicity is not None and profile.ethnicity != ethnicity:
            continue
        if gender is not None and profile.gender != gender:
            continue
        if location_type is not None and profile.location_type != location_type:
            continue
        if occupation_type is not None and profile.occupation_type != occupation_type:
            continue

        filtered_records.append(record)

    return filtered_records


def get_dataset_statistics(
    file_path: Path | str | None = None,
) -> dict[str, dict[str, int] | int]:
    """Get statistics about the loan approval dataset.

    Args:
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        Dictionary containing statistics about approvals, demographics, etc.
    """
    all_records = load_loan_approval_dataset(file_path)

    # Count approvals
    approval_counts = {"Approved": 0, "Rejected": 0}
    for record in all_records:
        approval_counts[record.approval_status] += 1

    # Count demographics
    ethnicity_counts: dict[str, int] = {}
    gender_counts: dict[str, int] = {}
    location_type_counts: dict[str, int] = {}
    occupation_type_counts: dict[str, int] = {}

    for record in all_records:
        profile = record.demographic_profile

        ethnicity_counts[profile.ethnicity] = (
            ethnicity_counts.get(profile.ethnicity, 0) + 1
        )
        gender_counts[profile.gender] = gender_counts.get(profile.gender, 0) + 1
        location_type_counts[profile.location_type] = (
            location_type_counts.get(profile.location_type, 0) + 1
        )
        occupation_type_counts[profile.occupation_type] = (
            occupation_type_counts.get(profile.occupation_type, 0) + 1
        )

    return {
        "total_records": len(all_records),
        "approvals": approval_counts,
        "ethnicity": ethnicity_counts,
        "gender": gender_counts,
        "location_type": location_type_counts,
        "occupation_type": occupation_type_counts,
    }
