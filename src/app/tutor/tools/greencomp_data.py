"""
GreenComp Data Loader
Loads and provides access to GreenComp learning outcomes and methods CSV files
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Get project root directory (parent of welearn_mas)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@lru_cache(maxsize=1)
def load_greencomp_outcomes() -> pd.DataFrame:
    """
    Load GreenComp learning outcomes CSV

    Returns:
        DataFrame with columns: Competence area, Competence, Learning outcome,
        KSA coverage, Observable, Measurable, Evaluation Methods
    """
    csv_path = PROJECT_ROOT / "GreenComp-LearningOutcomes.csv"

    if not csv_path.exists():
        logger.warning(f"GreenComp outcomes CSV not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} GreenComp learning outcomes")
    return df


@lru_cache(maxsize=1)
def load_greencomp_methods() -> pd.DataFrame:
    """
    Load GreenComp learning methods CSV

    Returns:
        DataFrame with columns: Method_Name, Description, Validation_Source,
        Best_For_LO_Types
    """
    csv_path = PROJECT_ROOT / "GreenComp-LearningMethods.csv"

    if not csv_path.exists():
        logger.warning(f"GreenComp methods CSV not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} GreenComp learning methods")
    return df


def get_suitable_methods_for_competencies(competencies: List[str]) -> List[Dict]:
    """
    Get suitable learning methods for given GreenComp competencies

    Args:
        competencies: List of GreenComp competency codes (e.g., ["C4", "C5"])

    Returns:
        List of dicts with method info: {name, description, best_for}
    """
    methods_df = load_greencomp_methods()
    outcomes_df = load_greencomp_outcomes()

    if methods_df.empty or outcomes_df.empty:
        return []

    # Map competencies to learning outcome types
    # Filter outcomes by competencies and get their types
    competency_names = []
    for comp_code in competencies:
        # Extract competency names from outcomes based on competency area
        matching_outcomes = outcomes_df[
            outcomes_df["Competence"].str.contains(comp_code, case=False, na=False)
        ]
        if not matching_outcomes.empty:
            competency_names.extend(matching_outcomes["Competence"].unique().tolist())

    # Find methods that match these competency types
    suitable_methods = []
    for _, method in methods_df.iterrows():
        best_for = str(method.get("Best_For_LO_Types", ""))

        # Check if any competency name appears in the best_for field
        if any(
            comp_name.lower() in best_for.lower()
            for comp_name in competency_names
            if comp_name
        ):
            suitable_methods.append(
                {
                    "name": method.get("Method_Name", ""),
                    "description": method.get("Description", ""),
                    "best_for": best_for,
                }
            )

    return suitable_methods


def get_suitable_methods_for_outcome_text(outcome_text: str) -> List[Dict]:
    """
    Get suitable learning methods for a given learning outcome by matching keywords

    Args:
        outcome_text: Text of the learning outcome

    Returns:
        List of dicts with method info, sorted by relevance
    """
    methods_df = load_greencomp_methods()

    if methods_df.empty:
        return []

    # Keywords to match in learning outcome text
    keyword_mapping = {
        "analyze": [
            "Case Study Analysis",
            "Critical Analysis / Deconstruction",
            "Systems Mapping",
        ],
        "evaluate": [
            "Critical Analysis / Deconstruction",
            "Policy & Governance Analysis",
        ],
        "design": ["Futures Literacy Workshop", "Living Lab / Co-creation"],
        "demonstrate": [
            "Community Project (Participatory Action)",
            "Reflective Practice / Portfolio",
        ],
        "stakeholder": [
            "Stakeholder Simulation / Role-Play",
            "Living Lab / Co-creation",
        ],
        "systems": ["Systems Mapping", "Interdisciplinary Research Project"],
        "policy": ["Policy & Governance Analysis", "Advocacy Campaign"],
        "future": ["Futures Literacy Workshop", "Socio-Ecological Transition Planning"],
        "personal": ["Personal Action Plan", "Reflective Practice / Portfolio"],
        "community": [
            "Community Project (Participatory Action)",
            "Living Lab / Co-creation",
        ],
        "sustainability": ["Case Study Analysis", "Interdisciplinary Research Project"],
    }

    outcome_lower = outcome_text.lower()
    method_scores = {}

    # Score each method based on keyword matches
    for keyword, method_names in keyword_mapping.items():
        if keyword in outcome_lower:
            for method_name in method_names:
                method_scores[method_name] = method_scores.get(method_name, 0) + 1

    # Get method details and sort by score
    suitable_methods = []
    for method_name, score in sorted(
        method_scores.items(), key=lambda x: x[1], reverse=True
    ):
        method_row = methods_df[methods_df["Method_Name"] == method_name]
        if not method_row.empty:
            method = method_row.iloc[0]
            suitable_methods.append(
                {
                    "name": method.get("Method_Name", ""),
                    "description": method.get("Description", ""),
                    "best_for": method.get("Best_For_LO_Types", ""),
                    "relevance_score": score,
                }
            )

    return suitable_methods[:5]  # Return top 5


def get_greencomp_competency_details(competency_code: str) -> Optional[Dict]:
    """
    Get detailed information about a GreenComp competency

    Args:
        competency_code: Competency code (e.g., "C4", "Systems thinking")

    Returns:
        Dict with competency details or None if not found
    """
    outcomes_df = load_greencomp_outcomes()

    if outcomes_df.empty:
        return None

    # Try to find by code or name
    matching = outcomes_df[
        (outcomes_df["Competence"].str.contains(competency_code, case=False, na=False))
        | (
            outcomes_df["Competence area"].str.contains(
                competency_code, case=False, na=False
            )
        )
    ]

    if matching.empty:
        return None

    first_match = matching.iloc[0]
    return {
        "competence_area": first_match.get("Competence area", ""),
        "competence": first_match.get("Competence", ""),
        "sample_outcome": first_match.get("Learning outcome", ""),
        "evaluation_methods": first_match.get("Evaluation Methods", ""),
    }


def get_all_methods() -> List[str]:
    """Get list of all available learning method names"""
    methods_df = load_greencomp_methods()
    if methods_df.empty:
        return []
    return methods_df["Method_Name"].tolist()


def get_method_description(method_name: str) -> Optional[str]:
    """Get description of a specific learning method"""
    methods_df = load_greencomp_methods()
    if methods_df.empty:
        return None

    method = methods_df[methods_df["Method_Name"] == method_name]
    if method.empty:
        return None

    return method.iloc[0].get("Description", "")
