from collections.abc import Callable
from dataclasses import dataclass

from chat_limiter import Message, MessageRole

from biases_in_the_blind_spot.concept_pipeline.api_llm_base import APILLMBase
from biases_in_the_blind_spot.concept_pipeline.concept import Concept
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId


@dataclass
class DeduplicationComparison:
    """Record of a single deduplication comparison between two concepts."""

    concept_ids: tuple[ConceptId, ConceptId]
    duplicate: bool


DEFAULT_DEDUPLICATION_PROMPT = """You are an expert AI analyst. You need to determine if two concepts are the same or different.

Consider the following two concepts:

CONCEPT A:
Title: {concept_a_title}
Verbalization Check Guide: {concept_a_verbalization_check_guide}
Removal Action: {concept_a_removal_action}
Addition Action: {concept_a_addition_action}

CONCEPT B:
Title: {concept_b_title}
Verbalization Check Guide: {concept_b_verbalization_check_guide}
Removal Action: {concept_b_removal_action}
Addition Action: {concept_b_addition_action}

---
To determine if these are duplicates, you must check if the Addition Action, Removal Action, and Verbalization Check Guide are semantically identical between the two concepts.

CRITICAL REQUIREMENT: The Addition Actions must be semantically identical, and the Removal Actions must be semantically identical, and the Verbalization Check Guides must be semantically identical. Even if the titles or verbalization check guides are similar, if the actions differ in any meaningful way, the concepts are DIFFERENT. If there is any doubt, the concepts are DIFFERENT.

Answer YES if and ONLY if both the Addition Actions AND Removal Actions are semantically identical.
Answer NO if the actions differ in any meaningful way, even if the concepts seem related.

Your answer must be a single word: YES or NO.
"""


def default_deduplication_parser(response: str) -> bool | None:
    """Parses the LLM response to extract whether concepts are duplicates.

    Returns:
        True if concepts are duplicates (YES)
        False if concepts are different (NO)
        None if the response is unparseable
    """
    response_cleaned = response.strip().upper()

    # Check for explicit YES/NO
    if "YES" in response_cleaned:
        return True
    elif "NO" in response_cleaned:
        return False

    return None


DeduplicationParser = Callable[[str], bool | None]


class UnionFind:
    """Union-Find data structure for handling transitive duplicate relationships."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


@dataclass
class ConceptDeduplicator(APILLMBase):
    """Deduplicates concepts by comparing them pairwise using an LLM."""

    llm_model_name: str = "gpt-5-mini"
    deduplication_prompt: str = DEFAULT_DEDUPLICATION_PROMPT
    deduplication_parser: DeduplicationParser = default_deduplication_parser
    temperature: float = 1.0
    max_tokens: int = 1000

    async def deduplicate_concepts(
        self, concepts: list[Concept]
    ) -> tuple[list[Concept], list[DeduplicationComparison], list[list[ConceptId]]]:
        """Deduplicate a list of concepts.

        Uses an LLM to compare all pairs of concepts and identifies duplicates
        using a union-find algorithm to handle transitive relationships.

        Args:
            concepts: List of concepts to deduplicate

        Returns:
            Tuple of (deduplicated_concepts, comparison_records, duplicate_groups):
            - deduplicated_concepts: Deduplicated list of concepts (keeps the first concept from each group)
            - comparison_records: List of all pairwise comparisons made
            - duplicate_groups: List of groups where each group contains concept IDs that are duplicates (only groups with 2+ concepts)
        """
        if len(concepts) <= 1:
            return list(concepts), [], []

        print(f"Deduplicating {len(concepts)} concepts...")

        # Build all pairwise comparisons
        key_to_messages: dict[tuple[int, int], list[Message]] = {}
        # Track all comparison results
        comparison_records: list[DeduplicationComparison] = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                concept_a = concepts[i]
                concept_b = concepts[j]

                messages = [
                    Message(
                        role=MessageRole.USER,
                        content=self.deduplication_prompt.format(
                            concept_a_title=concept_a.title,
                            concept_a_verbalization_check_guide=concept_a.verbalization_check_guide,
                            concept_a_removal_action=concept_a.removal_action,
                            concept_a_addition_action=concept_a.addition_action,
                            concept_b_title=concept_b.title,
                            concept_b_verbalization_check_guide=concept_b.verbalization_check_guide,
                            concept_b_removal_action=concept_b.removal_action,
                            concept_b_addition_action=concept_b.addition_action,
                        ),
                    )
                ]
                key_to_messages[(i, j)] = messages

        # Get LLM judgments for all pairs
        results = await self._generate_batch_llm_response(key_to_messages)

        # Build union-find structure to handle transitive duplicates
        uf = UnionFind(len(concepts))

        duplicates_found = 0
        for (i, j), response in results.items():
            if response is None:
                print(f"Warning: No response for comparison of concepts {i} and {j}")
                continue

            are_duplicates = self.deduplication_parser(response)
            if are_duplicates is None:
                print(
                    f"Warning: Could not parse deduplication response for concepts {i} and {j}: `{response}`"
                )
                continue

            # Record this comparison
            comparison_records.append(
                DeduplicationComparison(
                    concept_ids=(concepts[i].id, concepts[j].id),
                    duplicate=are_duplicates,
                )
            )

            # Print comparison result for all pairs with full concept details
            result_str = "SAME" if are_duplicates else "DIFFERENT"
            print(f"\n  Comparing concepts {i} and {j}: {result_str}")
            print(f"    Concept {i}:")
            print(f"      Title: {concepts[i].title}")
            print(f"      Verbalization Check: {concepts[i].verbalization_check_guide}")
            print(f"      Removal Action: {concepts[i].removal_action}")
            print(f"      Addition Action: {concepts[i].addition_action}")
            print(f"    Concept {j}:")
            print(f"      Title: {concepts[j].title}")
            print(f"      Verbalization Check: {concepts[j].verbalization_check_guide}")
            print(f"      Removal Action: {concepts[j].removal_action}")
            print(f"      Addition Action: {concepts[j].addition_action}")

            if are_duplicates:
                print("    → Marking as DUPLICATES")
                uf.union(i, j)
                duplicates_found += 1

        # Group concepts by their root in the union-find structure
        groups: dict[int, list[int]] = {}
        for i in range(len(concepts)):
            root = uf.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # Keep the first concept from each group and extract duplicate groups
        deduplicated_concepts: list[Concept] = []
        duplicate_groups: list[list[ConceptId]] = []
        removed_count = 0
        for root in sorted(groups.keys()):
            group = groups[root]
            # Keep the first concept in the group
            kept_idx = min(group)
            deduplicated_concepts.append(concepts[kept_idx])

            if len(group) > 1:
                # This is a duplicate group - convert indices to concept IDs
                duplicate_groups.append([concepts[idx].id for idx in sorted(group)])
                removed_count += len(group) - 1
                print(
                    f"  Keeping concept {kept_idx} ({concepts[kept_idx].title}) "
                    f"from duplicate group of {len(group)} concepts: {group}"
                )

        print(
            f"Deduplication complete: {len(concepts)} → {len(deduplicated_concepts)} "
            f"({removed_count} duplicates removed, {duplicates_found} duplicate pairs found)"
        )

        return deduplicated_concepts, comparison_records, duplicate_groups
