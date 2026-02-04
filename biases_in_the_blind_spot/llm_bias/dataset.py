import random

import pandas as pd


def balanced_downsample(
    df: pd.DataFrame, n_samples: int, random_seed: int = 42
) -> pd.DataFrame:
    """
    Downsample the dataset while maintaining balance across resume content.

    Args:
        df: DataFrame containing the dataset
        n_samples: Number of unique resume contents to sample
        random_seed: Random seed for reproducibility

    Returns:
        Downsampled DataFrame that maintains demographic variations for each resume
    """
    # Get unique resume strings
    unique_resumes = df["Resume_str"].unique()

    if n_samples > len(unique_resumes):
        print(
            f"Warning: Requested sample size ({n_samples}) is larger than unique resumes ({len(unique_resumes)}). Using all unique resumes."
        )
        return df

    # Randomly sample n_samples unique resumes
    sampled_resumes = random.sample(list(unique_resumes), n_samples)

    # Get all rows that contain these resume strings
    balanced_sample = df[df["Resume_str"].isin(sampled_resumes)]

    print(f"Downsampled to {len(sampled_resumes)} unique resumes")
    print(
        f"Total samples after maintaining demographic variations: {len(balanced_sample)}"
    )

    return balanced_sample  # type: ignore
