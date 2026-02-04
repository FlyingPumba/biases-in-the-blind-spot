import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_save_or_plot(monkeypatch):
    """Avoid disk writes and plotting side effects in tests."""
    no_op = lambda *args, **kwargs: None
    targets = [
        "biases_in_the_blind_spot.concept_pipeline.baseline_verbalization.save_result",
        "biases_in_the_blind_spot.concept_pipeline.baseline_verbalization.plot_concept_baseline_verbalization",
        "biases_in_the_blind_spot.concept_pipeline.variation_bias.save_result",
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.save_result",
        "biases_in_the_blind_spot.concept_pipeline.variation_verbalization.plot_variation_verbalization",
        "biases_in_the_blind_spot.concept_pipeline.pipeline_persistence.save_result",
        "biases_in_the_blind_spot.concept_pipeline.plotting.plot_bias_impact",
        "biases_in_the_blind_spot.concept_pipeline.variation_bias.plot_bias_impact",
    ]
    for target in targets:
        monkeypatch.setattr(target, no_op, raising=False)
