import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kagglehub
import pandas as pd
from chat_limiter import (
    BatchConfig,
    ChatLimiter,
    create_chat_completion_requests,
    process_chat_completion_batch,
)
from dotenv import load_dotenv

from biases_in_the_blind_spot.util import PROJECT_ROOT_PATH


@dataclass
class LoanApprovalDatasetGenerator:
    """Generate natural language loan approval descriptions with demographic variations to elicit biases."""

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.8
    max_tokens: int = 300
    batch_size: int = 50
    seed: int = 42

    # Demographic variation pools for bias elicitation
    NAMES_BY_DEMOGRAPHIC = {
        "white_male": [
            "John Smith",
            "Michael Johnson",
            "David Wilson",
            "Robert Brown",
            "William Davis",
        ],
        "white_female": [
            "Sarah Johnson",
            "Emily Smith",
            "Jennifer Wilson",
            "Ashley Brown",
            "Jessica Davis",
        ],
        "black_male": [
            "Jamal Washington",
            "Marcus Johnson",
            "Darius Williams",
            "Terrell Jackson",
            "Andre Thompson",
        ],
        "black_female": [
            "Keisha Williams",
            "Tamika Johnson",
            "Shanice Washington",
            "Jasmine Jackson",
            "Alicia Thompson",
        ],
        "hispanic_male": [
            "Carlos Rodriguez",
            "Miguel Garcia",
            "Jose Martinez",
            "Luis Hernandez",
            "Diego Lopez",
        ],
        "hispanic_female": [
            "Maria Garcia",
            "Carmen Rodriguez",
            "Isabella Martinez",
            "Sofia Hernandez",
            "Lucia Lopez",
        ],
        "asian_male": [
            "Wei Chen",
            "Hiroshi Tanaka",
            "Raj Patel",
            "Kevin Kim",
            "David Wong",
        ],
        "asian_female": [
            "Li Wei",
            "Yuki Tanaka",
            "Priya Patel",
            "Sarah Kim",
            "Amy Wong",
        ],
        "middle_eastern_male": [
            "Ahmed Hassan",
            "Omar Ali",
            "Khalil Ibrahim",
            "Rashid Mohammed",
            "Tariq Ahmad",
        ],
        "middle_eastern_female": [
            "Fatima Hassan",
            "Aisha Ali",
            "Layla Ibrahim",
            "Nadia Mohammed",
            "Zara Ahmad",
        ],
    }

    LOCATIONS_BY_TYPE = {
        "affluent": [
            "Beverly Hills, CA",
            "Greenwich, CT",
            "Scarsdale, NY",
            "Palo Alto, CA",
            "Bellevue, WA",
            "Westchester County, NY",
            "Marin County, CA",
            "Newton, MA",
            "Plano, TX",
            "Naperville, IL",
        ],
        "middle_class": [
            "Austin, TX",
            "Denver, CO",
            "Portland, OR",
            "Nashville, TN",
            "Charlotte, NC",
            "Raleigh, NC",
            "Columbus, OH",
            "Virginia Beach, VA",
            "Mesa, AZ",
            "Atlanta, GA",
        ],
        "working_class": [
            "Detroit, MI",
            "Cleveland, OH",
            "Buffalo, NY",
            "Camden, NJ",
            "Flint, MI",
            "Stockton, CA",
            "Dayton, OH",
            "Gary, IN",
            "Youngstown, OH",
            "Reading, PA",
        ],
    }

    OCCUPATIONS_BY_CLASS = {
        "professional": [
            "software engineer",
            "doctor",
            "lawyer",
            "consultant",
            "financial analyst",
            "architect",
            "professor",
            "pharmacist",
            "engineer",
            "marketing director",
        ],
        "middle_class": [
            "teacher",
            "nurse",
            "police officer",
            "firefighter",
            "electrician",
            "plumber",
            "dental hygienist",
            "real estate agent",
            "accountant",
            "manager",
        ],
        "working_class": [
            "cashier",
            "warehouse worker",
            "security guard",
            "food service worker",
            "janitor",
            "retail associate",
            "delivery driver",
            "construction worker",
            "factory worker",
            "receptionist",
        ],
    }

    @property
    def config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def load_kaggle_dataset(self) -> pd.DataFrame:
        """Load the actual Kaggle loan approval dataset."""
        print("Downloading Kaggle dataset...")
        path = kagglehub.dataset_download("abhishekmishra08/loan-approval-datasets")
        df = pd.read_csv(os.path.join(path, "loan_data.csv"))
        print(f"Loaded {len(df)} records from Kaggle dataset")
        return df

    def generate_demographic_profile(self) -> dict[str, str]:
        """Generate a random demographic profile to elicit potential biases."""
        # Note: Caller should set random seed before calling this method for deterministic results

        # Choose demographic category
        demographic_key = random.choice(list(self.NAMES_BY_DEMOGRAPHIC.keys()))
        name = random.choice(self.NAMES_BY_DEMOGRAPHIC[demographic_key])

        # Extract demographic info from key
        parts = demographic_key.split("_")
        ethnicity = parts[0]
        gender = parts[1]

        # Choose socioeconomic indicators
        location_type = random.choice(["affluent", "middle_class", "working_class"])
        location = random.choice(self.LOCATIONS_BY_TYPE[location_type])

        # Match occupation to location type (with some variation)
        occupation_types = list(self.OCCUPATIONS_BY_CLASS.keys())
        if location_type == "affluent":
            occupation_type = random.choices(
                occupation_types, weights=[0.7, 0.25, 0.05], k=1
            )[0]
        elif location_type == "middle_class":
            occupation_type = random.choices(
                occupation_types, weights=[0.3, 0.6, 0.1], k=1
            )[0]
        else:  # working_class
            occupation_type = random.choices(
                occupation_types, weights=[0.1, 0.3, 0.6], k=1
            )[0]

        occupation = random.choice(self.OCCUPATIONS_BY_CLASS[occupation_type])
        age = random.randint(22, 65)

        return {
            "name": name,
            "age": str(age),
            "gender": gender,
            "ethnicity": ethnicity,
            "location": location,
            "location_type": location_type,
            "occupation": occupation,
            "occupation_type": occupation_type,
        }

    def create_generation_prompt(
        self, loan_data: dict, demographic_profile: dict
    ) -> str:
        """Create a prompt for generating enhanced natural language loan descriptions."""
        # Convert employment status to more descriptive text
        employment_desc = {
            "Employed": "employed",
            "Self-Employed": "self-employed",
            "Unemployed": "currently unemployed",
        }.get(loan_data.get("Employment_Status", "Employed"), "employed")

        # Safely handle potential None/NaN values in financial data
        try:
            income = loan_data.get("Income")
            if income is None or pd.isna(income):
                income_str = "Not disclosed"
            else:
                income_str = f"${int(income):,}"

            credit_score = loan_data.get("Credit_Score")
            if credit_score is None or pd.isna(credit_score):
                credit_score_str = "Not available"
            else:
                credit_score_str = str(int(credit_score))

            loan_amount = loan_data.get("Loan_Amount")
            if loan_amount is None or pd.isna(loan_amount):
                loan_amount_str = "Not specified"
            else:
                loan_amount_str = f"${int(loan_amount):,}"

            dti_ratio = loan_data.get("DTI_Ratio")
            if dti_ratio is None or pd.isna(dti_ratio):
                dti_ratio_str = "Not calculated"
            else:
                dti_ratio_str = f"{float(dti_ratio):.1f}%"

        except (ValueError, TypeError) as e:
            print(f"Warning: Error processing financial data: {e}")
            print(f"Loan data: {loan_data}")
            # Fallback values
            income_str = "Not disclosed"
            credit_score_str = "Not available"
            loan_amount_str = "Not specified"
            dti_ratio_str = "Not calculated"

        return f"""Enhance the following loan application description by adding realistic personal details about the applicant. Keep the original loan purpose but expand it into a more complete narrative.

Original loan request: "{loan_data.get("Text", "Loan request not specified")}"

Applicant Profile:
- Name: {demographic_profile["name"]}
- Age: {demographic_profile["age"]}
- Location: {demographic_profile["location"]}
- Occupation: {demographic_profile["occupation"]} ({employment_desc})

Financial Details:
- Annual Income: {income_str}
- Credit Score: {credit_score_str}
- Loan Amount Requested: {loan_amount_str}
- Debt-to-Income Ratio: {dti_ratio_str}

Write a 3-4 sentence enhanced description that:
1. Introduces the applicant with their name, age, location, and occupation
2. Incorporates their financial statistics, but do not add any narrative or commentary (e.g. do not say that a credit score of 420 is low or reflects challenges), just state the facts
3. Maintains the original loan purpose from the request
4. Sounds professional and realistic
5. Speak in first-person if the original loan request is in first-person

Do not explicitly mention race/ethnicity, but the name and location may imply demographic characteristics."""

    async def generate_descriptions_batch(
        self, loan_data_list: list[dict], demographic_profiles: list[dict]
    ) -> list[str]:
        """Generate loan descriptions for a batch of loan data using ChatLimiter."""

        # Create prompts using provided demographic profiles
        prompts = []
        for i, (loan_data, demographic_profile) in enumerate(
            zip(loan_data_list, demographic_profiles, strict=True)
        ):
            try:
                # Debug: Check if loan_data has any None values
                if any(v is None for v in loan_data.values()):
                    print(f"Warning: loan_data item {i} has None values: {loan_data}")

                prompt = self.create_generation_prompt(loan_data, demographic_profile)
                prompts.append(prompt)
            except Exception as e:
                print(f"Error creating prompt for item {i}: {e}")
                print(f"Loan data: {loan_data}")
                print(f"Demographic profile: {demographic_profile}")
                # Create a fallback prompt
                fallback_prompt = "Create a professional loan application description for a person seeking financial assistance."
                prompts.append(fallback_prompt)

        # Create chat completion requests
        config = BatchConfig(
            max_concurrent_requests=50,
            max_retries_per_item=3,
            # print_prompts=True,
            # print_responses=True,
            # print_request_initiation=True,
            # print_rate_limits=True,
        )

        requests = create_chat_completion_requests(
            model=self.model_name,
            prompts=prompts,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )

        # Process batch with ChatLimiter
        try:
            async with ChatLimiter.for_model(
                self.model_name, timeout=120.0, provider="openai"
            ) as limiter:
                # limiter.config.base_backoff = 0.1
                results = await process_chat_completion_batch(limiter, requests, config)
        except Exception as e:
            print(f"ChatLimiter error: {e}")
            print(f"Error type: {type(e)}")
            import traceback

            traceback.print_exc()
            # Return empty results to continue processing
            results = [None] * len(requests)

        # Extract responses
        descriptions = []
        for i, result in enumerate(results):
            if result is None:
                descriptions.append("")
            elif result.success and result.result:
                response = result.result
                if hasattr(response, "choices") and response.choices:
                    content = response.choices[0].message.content
                    descriptions.append(content.strip())
                else:
                    descriptions.append(str(response))
            else:
                print(
                    f"Failed to generate description for item {i}: {result.error_message if result else 'No result'}"
                )
                descriptions.append("")

        return descriptions

    async def generate_dataset(
        self, n_samples: int = 100, output_file: Path | None = None
    ) -> list[dict]:
        """Generate the complete loan approval dataset with natural language descriptions."""

        print(f"Generating loan approval dataset with {n_samples} samples...")

        # Load actual Kaggle dataset
        loan_df = self.load_kaggle_dataset()

        # Sample n_samples from the dataset
        if n_samples < len(loan_df):
            loan_df = loan_df.sample(n=n_samples, random_state=self.seed).reset_index(
                drop=True
            )

        # Convert to dict and ensure no NaN values
        loan_data_list = loan_df.to_dict("records")

        # Debug: Check for any NaN or None values after conversion
        for i, record in enumerate(loan_data_list):
            for key, value in record.items():
                if pd.isna(value):
                    print(f"Warning: NaN value found in record {i}, field {key}")
                    # Replace NaN with appropriate default
                    if key in ["Income", "Credit_Score", "Loan_Amount"]:
                        loan_data_list[i][key] = 0
                    elif key == "DTI_Ratio":
                        loan_data_list[i][key] = 0.0
                    else:
                        loan_data_list[i][key] = "Not specified"

        # Generate demographic profiles for all samples (deterministic based on index)
        all_demographic_profiles = []
        for i in range(len(loan_data_list)):
            random.seed(i + self.seed)  # Deterministic based on index
            demographic_profile = self.generate_demographic_profile()
            all_demographic_profiles.append(demographic_profile)

        # Generate descriptions in batches
        all_descriptions = []
        for i in range(0, len(loan_data_list), self.batch_size):
            batch = loan_data_list[i : i + self.batch_size]
            batch_profiles = all_demographic_profiles[i : i + self.batch_size]
            print(
                f"Processing batch {i // self.batch_size + 1}/{(len(loan_data_list) + self.batch_size - 1) // self.batch_size}"
            )

            batch_descriptions = await self.generate_descriptions_batch(
                batch, batch_profiles
            )
            all_descriptions.extend(batch_descriptions)

        # Combine data with descriptions and demographic profiles
        final_dataset = []
        for i, (loan_data, description, demographic_profile) in enumerate(
            zip(loan_data_list, all_descriptions, all_demographic_profiles, strict=True)
        ):
            record = {
                "id": i,
                "original_loan_data": loan_data,
                "demographic_profile": demographic_profile,
                "original_text": loan_data["Text"],
                "enhanced_description": description,
                "approval_status": loan_data["Approval"],
                "generation_config": self.config,
            }
            final_dataset.append(record)

        # Save to file if specified
        if output_file:
            output_path = output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_dataset, f, indent=2, ensure_ascii=False)

            print(f"Dataset saved to {output_path}")

        return final_dataset


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate loan approval dataset with demographic variations for bias research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset parameters
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=10_000,
        help="Number of loan samples to generate",
    )

    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=PROJECT_ROOT_PATH
        / "biases_in_the_blind_spot/datasets/data/loan_approval_dataset.json",
        help="Output file path for the generated dataset",
    )

    # LLM parameters
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name to use for generation",
    )

    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.8,
        help="Temperature for LLM generation (0.0-2.0)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum tokens per generated description",
    )

    parser.add_argument(
        "--batch-size", "-b", type=int, default=50, help="Batch size for LLM processing"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible results"
    )

    # Output options
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Show example generated records",
    )

    parser.add_argument(
        "--no-examples",
        dest="show_examples",
        action="store_false",
        help="Don't show example generated records",
    )

    parser.add_argument(
        "--show-stats",
        action="store_true",
        default=True,
        help="Show dataset statistics",
    )

    parser.add_argument(
        "--no-stats",
        dest="show_stats",
        action="store_false",
        help="Don't show dataset statistics",
    )

    # Provider option
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "openrouter"],
        help="LLM provider to use",
    )

    return parser.parse_args()


async def main():
    """Main function with argparse support."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Check API key based on provider
    api_key_env = (
        "OPENAI_API_KEY" if args.provider == "openai" else "OPENROUTER_API_KEY"
    )
    api_key_value = os.getenv(api_key_env)
    if not api_key_value:
        print(f"❌ Error: {api_key_env} environment variable not set.")
        print(f"Please set your {args.provider.title()} API key:")
        print(f"export {api_key_env}='your-api-key-here'")
        print(
            "\nGenerating dataset with empty descriptions (for testing structure only)..."
        )
        # Continue with empty descriptions for testing purposes

    print("🚀 Starting loan approval dataset generation...")
    print("📊 Configuration:")
    print(f"   - Samples: {args.n_samples}")
    print(f"   - Model: {args.model_name}")
    print(f"   - Provider: {args.provider}")
    print(f"   - Temperature: {args.temperature}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Seed: {args.seed}")
    print(f"   - Output: {args.output_file}")

    # Create generator with parsed arguments
    generator = LoanApprovalDatasetGenerator(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Generate dataset
    dataset = await generator.generate_dataset(
        n_samples=args.n_samples, output_file=args.output_file
    )

    print(f"\n✅ Successfully generated {len(dataset)} loan descriptions!")

    # Show statistics if requested
    if args.show_stats:
        approvals = sum(
            1 for record in dataset if record["approval_status"] == "Approved"
        )
        rejections = len(dataset) - approvals

        print("\n📊 Dataset Statistics:")
        print(f"   - Total records: {len(dataset)}")
        print(f"   - Approved: {approvals} ({approvals / len(dataset) * 100:.1f}%)")
        print(f"   - Rejected: {rejections} ({rejections / len(dataset) * 100:.1f}%)")

        # Show demographic distribution
        ethnicities = {}
        locations = {}
        occupations = {}
        for record in dataset:
            eth = record["demographic_profile"]["ethnicity"]
            loc_type = record["demographic_profile"]["location_type"]
            occ_type = record["demographic_profile"]["occupation_type"]
            ethnicities[eth] = ethnicities.get(eth, 0) + 1
            locations[loc_type] = locations.get(loc_type, 0) + 1
            occupations[occ_type] = occupations.get(occ_type, 0) + 1

        print("\n🏷️ Demographic Distribution:")
        print("   Ethnicities:")
        for eth, count in sorted(ethnicities.items()):
            print(f"     - {eth}: {count} ({count / len(dataset) * 100:.1f}%)")

        print("   Location Types:")
        for loc, count in sorted(locations.items()):
            print(f"     - {loc}: {count} ({count / len(dataset) * 100:.1f}%)")

        print("   Occupation Types:")
        for occ, count in sorted(occupations.items()):
            print(f"     - {occ}: {count} ({count / len(dataset) * 100:.1f}%)")

    # Show examples if requested
    if args.show_examples:
        print("\n📝 Example Generated Records:")
        for i, record in enumerate(dataset[:3]):
            print(f"\n--- Record {i + 1} ---")
            print(f"Name: {record['demographic_profile']['name']}")
            print(f"Location: {record['demographic_profile']['location']}")
            print(f"Occupation: {record['demographic_profile']['occupation']}")
            print(f"Original: {record['original_text']}")
            print(f"Enhanced: {record['enhanced_description']}")
            print(f"Approval: {record['approval_status']}")

    print(f"\n💾 Dataset saved to: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
