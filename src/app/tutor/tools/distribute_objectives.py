"""
Objective distribution tool - Deterministic distribution of objectives across sessions (Frugal AI)
"""

import logging
from typing import List

from src.app.baml_client.async_client import types

logger = logging.getLogger(__name__)


def distribute_objectives(
    objectives: List[types.LearningObjective], metadata: types.CourseMetadata
) -> List[types.SessionPlan]:
    """
    Distribute learning objectives evenly across sessions

    Simple algorithm for prototype:
    - Divide objectives evenly
    - Distribute remainder to first sessions
    - In future: pedagogical sequencing (foundational concepts first)

    Args:
        objectives: List of learning objectives to distribute
        metadata: Course metadata with session info

    Returns:
        List of SessionPlan objects with assigned objectives
    """
    num_sessions = metadata.num_sessions
    num_objectives = len(objectives)

    logger.info(
        f"Distributing {num_objectives} objectives across {num_sessions} sessions"
    )

    # Calculate distribution
    objectives_per_session = num_objectives // num_sessions
    remainder = num_objectives % num_sessions

    sessions = []
    current_objective_idx = 0

    for session_num in range(1, num_sessions + 1):
        # Calculate how many objectives for this session
        count = objectives_per_session + (1 if session_num <= remainder else 0)

        # Assign next 'count' objectives
        session_objectives = [
            objectives[current_objective_idx + i].number for i in range(count)
        ]
        current_objective_idx += count

        # Create session plan
        session = types.SessionPlan(
            session_number=session_num,
            objectives=session_objectives,
            type=metadata.session_type,
            mode=metadata.session_mode,
            duration=metadata.session_duration,
            class_size=metadata.class_size,
        )

        sessions.append(session)

        logger.debug(
            f"Session {session_num}: {len(session_objectives)} objectives "
            f"(numbers: {session_objectives})"
        )

    logger.info(f"Created {len(sessions)} session plans")
    return sessions
