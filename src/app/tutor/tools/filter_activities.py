"""
Activity filtering tool - Deterministic filtering of activity templates (Frugal AI)
"""

import json
import logging
from pathlib import Path
from typing import List

from src.app.baml_client.async_client import types

logger = logging.getLogger(__name__)

# Cached activity templates (CAG not RAG)
_activity_templates: List[types.ActivityTemplate] = []


def load_activity_templates(filepath: str = None) -> List[types.ActivityTemplate]:
    """
    Load activity templates from JSON file

    Args:
        filepath: Path to activity templates JSON file

    Returns:
        List of ActivityTemplate objects
    """
    global _activity_templates

    if filepath is None:
        # Use default path
        filepath = Path(__file__).parent.parent / "data" / "activity_templates.json"

    if not Path(filepath).exists():
        logger.warning(
            f"Activity templates file not found: {filepath}. Using mock data."
        )
        _activity_templates = _get_mock_templates()
        return _activity_templates

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    _activity_templates = [types.ActivityTemplate(**template) for template in data]
    logger.info(f"Loaded {len(_activity_templates)} activity templates")

    return _activity_templates


def _get_mock_templates() -> List[types.ActivityTemplate]:
    """Get mock activity templates for testing"""
    return [
        types.ActivityTemplate(
            id="act_001",
            name="Débat structuré",
            description="Débat en groupes sur des questions controversées",
            compatible_types=["CM", "TD", "Séminaire"],
            compatible_modes=[types.SessionMode.PRESENTIEL, types.SessionMode.HYBRIDE],
            min_duration=1.5,
            max_duration=3.0,
            min_size=15,
            max_size=100,
            estimated_duration=2.0,
        ),
        types.ActivityTemplate(
            id="act_002",
            name="Étude de cas collaborative",
            description="Analyse de cas en petits groupes",
            compatible_types=["TD", "TP"],
            compatible_modes=[
                types.SessionMode.PRESENTIEL,
                types.SessionMode.DISTANCIEL,
                types.SessionMode.HYBRIDE,
            ],
            min_duration=1.0,
            max_duration=2.5,
            min_size=10,
            max_size=40,
            estimated_duration=1.5,
        ),
        types.ActivityTemplate(
            id="act_003",
            name="Présentation magistrale interactive",
            description="Cours avec moments d'interaction et questions",
            compatible_types=["CM"],
            compatible_modes=[
                types.SessionMode.PRESENTIEL,
                types.SessionMode.DISTANCIEL,
                types.SessionMode.HYBRIDE,
            ],
            min_duration=1.0,
            max_duration=4.0,
            min_size=20,
            max_size=300,
            estimated_duration=2.0,
        ),
        types.ActivityTemplate(
            id="act_004",
            name="Atelier pratique",
            description="Exercices pratiques individuels ou en binôme",
            compatible_types=["TP", "TD"],
            compatible_modes=[types.SessionMode.PRESENTIEL],
            min_duration=1.5,
            max_duration=3.0,
            min_size=8,
            max_size=30,
            estimated_duration=2.0,
        ),
        types.ActivityTemplate(
            id="act_005",
            name="Quiz formatif",
            description="Quiz interactif pour vérifier la compréhension",
            compatible_types=["CM", "TD", "TP"],
            compatible_modes=[
                types.SessionMode.PRESENTIEL,
                types.SessionMode.DISTANCIEL,
                types.SessionMode.HYBRIDE,
            ],
            min_duration=0.5,
            max_duration=1.5,
            min_size=5,
            max_size=200,
            estimated_duration=0.5,
        ),
    ]


def filter_activities(
    session_type: str, session_mode: types.SessionMode, duration: float, class_size: int
) -> List[types.ActivityTemplate]:
    """
    Filter activity templates based on session constraints (deterministic, no LLM)

    Args:
        session_type: Type of session (CM, TD, TP, Séminaire, etc.)
        session_mode: Mode of session (Présentiel, Distanciel, Hybride)
        duration: Session duration in hours
        class_size: Number of students

    Returns:
        List of compatible ActivityTemplate objects
    """
    global _activity_templates

    # Load templates if not already loaded
    if not _activity_templates:
        load_activity_templates()

    logger.info(
        f"Filtering activities: type={session_type}, mode={session_mode}, "
        f"duration={duration}h, size={class_size}"
    )

    filtered = []

    for activity in _activity_templates:
        # Check session type compatibility
        if session_type not in activity.compatible_types:
            continue

        # Check mode compatibility
        if session_mode not in activity.compatible_modes:
            continue

        # Check duration constraints
        if not (activity.min_duration <= duration <= activity.max_duration):
            continue

        # Check class size constraints
        if not (activity.min_size <= class_size <= activity.max_size):
            continue

        filtered.append(activity)

    logger.info(f"Filtered to {len(filtered)} compatible activities")
    return filtered
