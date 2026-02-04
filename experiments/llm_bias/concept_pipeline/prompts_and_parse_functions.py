from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
import pandas as pd


from biases_in_the_blind_spot.llm_bias.dataset import balanced_downsample
from biases_in_the_blind_spot.llm_bias.hiring_bias_prompts import (
    fix_inconsistencies_in_resume,
    remove_resume_title,
)
from biases_in_the_blind_spot.util import LLM_BIAS_DATA_PATH


def parse_cot_response(_: InputId, response: str) -> int | None:
    """Extract yes/no answer from Chain of Thought response."""
    response_lower = response.strip().lower()

    # Look for the final answer after "Answer:" marker
    if "answer:" in response_lower:
        final_answer = response_lower.split("answer:")[-1].strip()
        if final_answer.startswith("yes"):
            return 1
        elif final_answer.startswith("no"):
            return 0

    # Fallback for simple yes/no
    if response_lower.startswith("yes"):
        return 1
    if response_lower.startswith("no"):
        return 0

    return None


def load_resumes_with_pronouns(downsample=True) -> list[str]:
    dataset_path = LLM_BIAS_DATA_PATH / "resume/selected_cats_resumes.csv"
    df = pd.read_csv(dataset_path)

    if downsample:
        df = balanced_downsample(df, 150, random_seed=42)

    resumes = []
    for i, row in df.iterrows():
        base_resume = row["Resume_str"]
        name = row["First_name"] + " " + row["Last_name"]
        gender = row["Gender"]
        race = row["Race"]
        pronouns = "(He/him)" if gender == "Male" else "(She/her)"
        email = f"{row['First_name'].lower()}.{row['Last_name'].lower()}@gmail.com"  # type: ignore

        assert race.lower() in ["black", "white"], f"Unknown race: {race}"  # type: ignore
        assert gender.lower() in ["male", "female"], f"Unknown gender: {gender}"  # type: ignore

        base_resume = remove_resume_title(base_resume)
        base_resume = f"Name: {name} {pronouns}\nEmail: {email}\n\n" + base_resume
        job_category = row["Category"]

        base_resume = fix_inconsistencies_in_resume(base_resume, name, gender, race, pronouns, email, job_category)  # type: ignore
        resumes.append(base_resume)

    return resumes
