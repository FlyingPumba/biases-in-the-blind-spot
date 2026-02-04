"""Helper functions to load university admission dataset."""

import json
from pathlib import Path

from biases_in_the_blind_spot.util import PROJECT_ROOT_PATH, ensure_decompressed

from .data_classes import (
    DemographicProfile,
    GenerationConfig,
    OriginalStudentData,
    SATScores,
    UniversityAdmissionRecord,
)


def load_university_admission_dataset(
    file_path: Path | str | None = None,
) -> list[UniversityAdmissionRecord]:
    """Load the university admission dataset and return a list of UniversityAdmissionRecord dataclasses.

    Args:
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of UniversityAdmissionRecord objects.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if file_path is None:
        file_path = (
            PROJECT_ROOT_PATH
            / "biases_in_the_blind_spot/datasets/data/university_admission_dataset.json"
        )

    file_path = Path(file_path)
    file_path = ensure_decompressed(file_path)

    with open(file_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for item in raw_data:
        # Parse original student data
        orig_data = item["original_student_data"]
        original_student_data = OriginalStudentData(
            sex=orig_data["sex"],
            sat_v=orig_data["sat_v"],
            sat_m=orig_data["sat_m"],
            sat_sum=orig_data["sat_sum"],
            hs_gpa=orig_data["hs_gpa"],
            fy_gpa=orig_data["fy_gpa"],
        )

        # Parse SAT scores
        sat_data = item["sat_scores"]
        sat_scores = SATScores(
            verbal=sat_data["verbal"],
            math=sat_data["math"],
            total=sat_data["total"],
        )

        # Parse demographic profile
        demo_data = item["demographic_profile"]
        demographic_profile = DemographicProfile(
            name=demo_data["name"],
            gender=demo_data["gender"],
            ethnicity=demo_data["ethnicity"],
            high_school=demo_data["high_school"],
            high_school_type=demo_data["high_school_type"],
            first_generation=demo_data["first_generation"],
            legacy=demo_data["legacy"],
            extracurriculars=demo_data["extracurriculars"],
            intended_major=demo_data["intended_major"],
            major_category=demo_data["major_category"],
            recommendation_sources=demo_data["recommendation_sources"],
            awards=demo_data["awards"],
            essay_theme=demo_data["essay_theme"],
            family_income_bracket=demo_data["family_income_bracket"],
            family_income_range=demo_data["family_income_range"],
            expected_family_contribution=demo_data["expected_family_contribution"],
            financial_aid_status=demo_data["financial_aid_status"],
            financial_aid_description=demo_data["financial_aid_description"],
            parent_education_level=demo_data["parent_education_level"],
            parent_education_description=demo_data["parent_education_description"],
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
        record = UniversityAdmissionRecord(
            id=item["id"],
            original_student_data=original_student_data,
            sat_scores=sat_scores,
            demographic_profile=demographic_profile,
            application_summary=item["application_summary"],
            generation_config=generation_config,
        )

        records.append(record)

    return records


def load_first_generation_students(
    file_path: Path | str | None = None,
) -> list[UniversityAdmissionRecord]:
    """Load only the first-generation college student records.

    Args:
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of UniversityAdmissionRecord objects where first_generation is True.
    """
    all_records = load_university_admission_dataset(file_path)
    return [
        record for record in all_records if record.demographic_profile.first_generation
    ]


def load_legacy_students(
    file_path: Path | str | None = None,
) -> list[UniversityAdmissionRecord]:
    """Load only the legacy applicant records.

    Args:
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of UniversityAdmissionRecord objects where legacy is True.
    """
    all_records = load_university_admission_dataset(file_path)
    return [record for record in all_records if record.demographic_profile.legacy]


def load_students_by_demographic(
    ethnicity: str | None = None,
    gender: str | None = None,
    high_school_type: str | None = None,
    major_category: str | None = None,
    file_path: Path | str | None = None,
) -> list[UniversityAdmissionRecord]:
    """Load student records filtered by demographic criteria.

    Args:
        ethnicity: Filter by ethnicity (e.g., "white", "black", "hispanic", "asian", "middle_eastern").
        gender: Filter by gender ("male" or "female").
        high_school_type: Filter by high school type ("elite_private", "elite_public",
            "affluent_suburban", "middle_class_suburban", "urban_public", "rural_public").
        major_category: Filter by major category ("stem", "humanities", "business", "arts", "pre_professional").
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of UniversityAdmissionRecord objects matching the demographic criteria.
    """
    all_records = load_university_admission_dataset(file_path)

    filtered_records = []
    for record in all_records:
        profile = record.demographic_profile

        # Apply filters
        if ethnicity is not None and profile.ethnicity != ethnicity:
            continue
        if gender is not None and profile.gender != gender:
            continue
        if (
            high_school_type is not None
            and profile.high_school_type != high_school_type
        ):
            continue
        if major_category is not None and profile.major_category != major_category:
            continue

        filtered_records.append(record)

    return filtered_records


def load_students_by_financial_status(
    income_bracket: str | None = None,
    financial_aid_status: str | None = None,
    parent_education_level: str | None = None,
    file_path: Path | str | None = None,
) -> list[UniversityAdmissionRecord]:
    """Load student records filtered by financial criteria.

    Args:
        income_bracket: Filter by family income bracket ("high", "upper_middle", "middle",
            "lower_middle", "low").
        financial_aid_status: Filter by financial aid status ("full_need", "partial_need",
            "merit_only", "no_aid").
        parent_education_level: Filter by parent education level ("graduate_degree",
            "bachelors_degree", "some_college", "high_school", "less_than_high_school").
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of UniversityAdmissionRecord objects matching the financial criteria.
    """
    all_records = load_university_admission_dataset(file_path)

    filtered_records = []
    for record in all_records:
        profile = record.demographic_profile

        # Apply filters
        if (
            income_bracket is not None
            and profile.family_income_bracket != income_bracket
        ):
            continue
        if (
            financial_aid_status is not None
            and profile.financial_aid_status != financial_aid_status
        ):
            continue
        if (
            parent_education_level is not None
            and profile.parent_education_level != parent_education_level
        ):
            continue

        filtered_records.append(record)

    return filtered_records


def load_students_by_academics(
    min_gpa: float | None = None,
    max_gpa: float | None = None,
    min_sat: int | None = None,
    max_sat: int | None = None,
    file_path: Path | str | None = None,
) -> list[UniversityAdmissionRecord]:
    """Load student records filtered by academic criteria.

    Args:
        min_gpa: Minimum high school GPA (inclusive).
        max_gpa: Maximum high school GPA (inclusive).
        min_sat: Minimum total SAT score (inclusive).
        max_sat: Maximum total SAT score (inclusive).
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        List of UniversityAdmissionRecord objects matching the academic criteria.
    """
    all_records = load_university_admission_dataset(file_path)

    filtered_records = []
    for record in all_records:
        gpa = record.original_student_data.hs_gpa
        sat = record.sat_scores.total

        # Apply filters
        if min_gpa is not None and gpa < min_gpa:
            continue
        if max_gpa is not None and gpa > max_gpa:
            continue
        if min_sat is not None and sat < min_sat:
            continue
        if max_sat is not None and sat > max_sat:
            continue

        filtered_records.append(record)

    return filtered_records


def get_dataset_statistics(
    file_path: Path | str | None = None,
) -> dict[str, dict[str, int] | int | float]:
    """Get statistics about the university admission dataset.

    Args:
        file_path: Path to the JSON dataset file. If None, uses the default location.

    Returns:
        Dictionary containing statistics about demographics, academics, financials, etc.
    """
    all_records = load_university_admission_dataset(file_path)

    # Count demographics
    ethnicity_counts: dict[str, int] = {}
    gender_counts: dict[str, int] = {}
    high_school_type_counts: dict[str, int] = {}
    major_category_counts: dict[str, int] = {}
    income_bracket_counts: dict[str, int] = {}
    financial_aid_counts: dict[str, int] = {}
    parent_education_counts: dict[str, int] = {}

    first_gen_count = 0
    legacy_count = 0
    total_gpa = 0.0
    total_sat = 0

    for record in all_records:
        profile = record.demographic_profile

        ethnicity_counts[profile.ethnicity] = (
            ethnicity_counts.get(profile.ethnicity, 0) + 1
        )
        gender_counts[profile.gender] = gender_counts.get(profile.gender, 0) + 1
        high_school_type_counts[profile.high_school_type] = (
            high_school_type_counts.get(profile.high_school_type, 0) + 1
        )
        major_category_counts[profile.major_category] = (
            major_category_counts.get(profile.major_category, 0) + 1
        )
        income_bracket_counts[profile.family_income_bracket] = (
            income_bracket_counts.get(profile.family_income_bracket, 0) + 1
        )
        financial_aid_counts[profile.financial_aid_status] = (
            financial_aid_counts.get(profile.financial_aid_status, 0) + 1
        )
        parent_education_counts[profile.parent_education_level] = (
            parent_education_counts.get(profile.parent_education_level, 0) + 1
        )

        if profile.first_generation:
            first_gen_count += 1
        if profile.legacy:
            legacy_count += 1

        total_gpa += record.original_student_data.hs_gpa
        total_sat += record.sat_scores.total

    n = len(all_records)
    return {
        "total_records": n,
        "ethnicity": ethnicity_counts,
        "gender": gender_counts,
        "high_school_type": high_school_type_counts,
        "major_category": major_category_counts,
        "income_bracket": income_bracket_counts,
        "financial_aid_status": financial_aid_counts,
        "parent_education": parent_education_counts,
        "first_generation_count": first_gen_count,
        "legacy_count": legacy_count,
        "average_gpa": total_gpa / n if n > 0 else 0.0,
        "average_sat": total_sat / n if n > 0 else 0,
    }
