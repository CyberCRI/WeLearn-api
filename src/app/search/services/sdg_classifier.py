from src.app.search.infra.sdg_classifier import get_sdg_predictions

# below this confidence the classifier's guess is noise
SDG_MIN_PREDICTION = 0.1
# the SDGs the search page shows for a query
MAX_QUERY_SDGS = 3


async def classify_sdgs(text: str) -> list[int]:
    """The SDGs `text` is about, most likely first."""
    predictions = await get_sdg_predictions(text)
    likely = [p for p in predictions if p.prediction > SDG_MIN_PREDICTION]
    likely.sort(key=lambda p: p.prediction, reverse=True)
    return [p.sdg.code for p in likely[:MAX_QUERY_SDGS]]
