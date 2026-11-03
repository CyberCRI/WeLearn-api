import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from src.app.baml_client.async_client import types

# The tools package re-exports functions named like their modules, so
# ``tools.filter_activities`` is the function: fetch the modules explicitly.
fa_module = importlib.import_module("src.app.tutor.services.tools.filter_activities")
do_module = importlib.import_module(
    "src.app.tutor.services.tools.distribute_objectives"
)
vf_module = importlib.import_module("src.app.tutor.services.tools.validate_format")
gc_module = importlib.import_module("src.app.tutor.services.tools.greencomp_data")


def make_metadata(num_sessions: int) -> types.CourseMetadata:
    return types.CourseMetadata(
        discipline="Biology",
        topic="Ecology",
        level="L2",
        num_sessions=num_sessions,
        session_duration=2.0,
        session_type="TD",
        session_mode=types.SessionMode.HYBRIDE,
        class_size=25,
        output_language="en",
    )


def make_objectives(n: int) -> list[types.LearningObjective]:
    return [types.LearningObjective(number=i, text=f"Obj {i}") for i in range(1, n + 1)]


class TestDistributeObjectives(unittest.TestCase):
    def test_even_distribution(self):
        sessions = do_module.distribute_objectives(make_objectives(6), make_metadata(3))

        self.assertEqual([s.session_number for s in sessions], [1, 2, 3])
        self.assertEqual([s.objectives for s in sessions], [[1, 2], [3, 4], [5, 6]])

    def test_remainder_goes_to_first_sessions(self):
        sessions = do_module.distribute_objectives(make_objectives(7), make_metadata(3))

        self.assertEqual([s.objectives for s in sessions], [[1, 2, 3], [4, 5], [6, 7]])

    def test_session_plans_copy_metadata(self):
        metadata = make_metadata(2)
        sessions = do_module.distribute_objectives(make_objectives(2), metadata)

        for session in sessions:
            self.assertIsInstance(session, types.SessionPlan)
            self.assertEqual(session.type, "TD")
            self.assertEqual(session.mode, types.SessionMode.HYBRIDE)
            self.assertEqual(session.duration, 2.0)
            self.assertEqual(session.class_size, 25)

    def test_more_sessions_than_objectives(self):
        sessions = do_module.distribute_objectives(make_objectives(2), make_metadata(4))

        self.assertEqual(len(sessions), 4)
        self.assertEqual([s.objectives for s in sessions], [[1], [2], [], []])

    def test_uses_objective_numbers_not_positions(self):
        objectives = [
            types.LearningObjective(number=10, text="a"),
            types.LearningObjective(number=20, text="b"),
        ]
        sessions = do_module.distribute_objectives(objectives, make_metadata(1))

        self.assertEqual(sessions[0].objectives, [10, 20])


class TestValidateFormat(unittest.TestCase):
    class Schema(BaseModel):
        name: str
        count: int

    def test_valid_content(self):
        result = vf_module.validate_format({"name": "x", "count": 3}, self.Schema)

        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])

    def test_invalid_content_lists_errors(self):
        result = vf_module.validate_format({"name": "x", "count": "nope"}, self.Schema)

        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("count", result.errors[0])

    def test_missing_fields(self):
        result = vf_module.validate_format({}, self.Schema)

        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 2)

    def test_unexpected_error_is_captured(self):
        class Exploding(BaseModel):
            def __init__(self, **data):
                raise RuntimeError("boom")

        result = vf_module.validate_format({"a": 1}, Exploding)

        self.assertFalse(result.valid)
        self.assertEqual(result.errors, ["boom"])


class TestFilterActivities(unittest.TestCase):
    def setUp(self):
        self._saved = fa_module._activity_templates
        fa_module._activity_templates = []

    def tearDown(self):
        fa_module._activity_templates = self._saved

    def test_load_missing_file_falls_back_to_mock_templates(self):
        templates = fa_module.load_activity_templates("/nonexistent/templates.json")

        self.assertEqual(len(templates), 5)
        self.assertEqual(templates[0].id, "act_001")
        self.assertIs(fa_module._activity_templates, templates)

    def test_load_default_path(self):
        # no data/activity_templates.json ships with the repo -> mock templates
        templates = fa_module.load_activity_templates()

        self.assertEqual(
            [t.id for t in templates],
            ["act_001", "act_002", "act_003", "act_004", "act_005"],
        )

    def test_load_from_json_file(self):
        data = [
            {
                "id": "custom_1",
                "name": "Fishbowl",
                "description": "Inner/outer circle discussion",
                "compatible_types": ["TD"],
                "compatible_modes": ["PRESENTIEL"],
                "min_duration": 1.0,
                "max_duration": 2.0,
                "min_size": 10,
                "max_size": 30,
                "estimated_duration": 1.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "templates.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            templates = fa_module.load_activity_templates(str(path))

        self.assertEqual(len(templates), 1)
        self.assertIsInstance(templates[0], types.ActivityTemplate)
        self.assertEqual(templates[0].id, "custom_1")
        self.assertEqual(templates[0].compatible_modes, [types.SessionMode.PRESENTIEL])

    def test_filter_lazily_loads_templates(self):
        with patch.object(
            fa_module,
            "load_activity_templates",
            wraps=fa_module.load_activity_templates,
        ) as mock_load:
            result = fa_module.filter_activities(
                "TD", types.SessionMode.PRESENTIEL, 2.0, 20
            )

        mock_load.assert_called_once_with()
        self.assertEqual([a.id for a in result], ["act_001", "act_002", "act_004"])

    def test_filter_does_not_reload_cached_templates(self):
        fa_module._activity_templates = fa_module._get_mock_templates()[:1]

        with patch.object(fa_module, "load_activity_templates") as mock_load:
            result = fa_module.filter_activities(
                "CM", types.SessionMode.PRESENTIEL, 2.0, 50
            )

        mock_load.assert_not_called()
        self.assertEqual([a.id for a in result], ["act_001"])

    def test_filter_by_mode(self):
        result = fa_module.filter_activities(
            "TD", types.SessionMode.DISTANCIEL, 2.0, 20
        )

        self.assertEqual([a.id for a in result], ["act_002"])

    def test_filter_by_type(self):
        result = fa_module.filter_activities(
            "Séminaire", types.SessionMode.HYBRIDE, 2.0, 50
        )

        self.assertEqual([a.id for a in result], ["act_001"])

    def test_filter_by_duration(self):
        result = fa_module.filter_activities("CM", types.SessionMode.HYBRIDE, 0.5, 50)

        self.assertEqual([a.id for a in result], ["act_005"])

    def test_filter_by_class_size(self):
        result = fa_module.filter_activities(
            "TD", types.SessionMode.PRESENTIEL, 2.0, 35
        )

        self.assertEqual([a.id for a in result], ["act_001", "act_002"])

    def test_filter_no_match(self):
        result = fa_module.filter_activities(
            "Unknown", types.SessionMode.PRESENTIEL, 2.0, 20
        )

        self.assertEqual(result, [])


OUTCOMES_CSV = """Competence area,Competence,Learning outcome,KSA coverage,Observable,Measurable,Evaluation Methods
Embracing complexity,Systems thinking,Maps a system,K,yes,yes,Concept map
Embracing complexity,Systems thinking,Explains feedback loops,S,yes,yes,Essay
Embracing complexity,Critical thinking,Assesses claims,S,yes,yes,Debate
Envisioning futures,Futures literacy,Imagines scenarios,A,yes,no,Portfolio
"""

METHODS_CSV = """Method_Name,Description,Validation_Source,Best_For_LO_Types
Systems Mapping,Draw causal loop diagrams,UNESCO,Systems thinking
Case Study Analysis,Analyse a real case,OECD,Critical thinking; Systems thinking
Futures Literacy Workshop,Imagine alternative futures,UNESCO,Futures literacy
Policy & Governance Analysis,Study public policies,EU,Political agency
"""


class TestGreenCompData(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        gc_module.load_greencomp_outcomes.cache_clear()
        gc_module.load_greencomp_methods.cache_clear()
        patcher = patch.object(gc_module, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        gc_module.load_greencomp_outcomes.cache_clear()
        gc_module.load_greencomp_methods.cache_clear()
        self._tmp.cleanup()

    def write_csvs(self, outcomes=True, methods=True):
        if outcomes:
            (self.root / "GreenComp-LearningOutcomes.csv").write_text(
                OUTCOMES_CSV, encoding="utf-8"
            )
        if methods:
            (self.root / "GreenComp-LearningMethods.csv").write_text(
                METHODS_CSV, encoding="utf-8"
            )

    # --- missing files -------------------------------------------------

    def test_loaders_return_empty_frames_when_files_missing(self):
        self.assertTrue(gc_module.load_greencomp_outcomes().empty)
        self.assertTrue(gc_module.load_greencomp_methods().empty)

    def test_helpers_degrade_gracefully_without_data(self):
        self.assertEqual(gc_module.get_suitable_methods_for_competencies(["x"]), [])
        self.assertEqual(gc_module.get_suitable_methods_for_outcome_text("analyze"), [])
        self.assertIsNone(gc_module.get_greencomp_competency_details("Systems"))
        self.assertEqual(gc_module.get_all_methods(), [])
        self.assertIsNone(gc_module.get_method_description("Systems Mapping"))

    def test_competencies_need_both_files(self):
        self.write_csvs(outcomes=False)

        self.assertEqual(
            gc_module.get_suitable_methods_for_competencies(["Systems thinking"]), []
        )

    # --- loaders -------------------------------------------------------

    def test_loaders_read_csvs_and_cache(self):
        self.write_csvs()

        outcomes = gc_module.load_greencomp_outcomes()
        methods = gc_module.load_greencomp_methods()

        self.assertEqual(len(outcomes), 4)
        self.assertEqual(len(methods), 4)
        self.assertIn("Evaluation Methods", outcomes.columns)
        self.assertIs(gc_module.load_greencomp_outcomes(), outcomes)
        self.assertIs(gc_module.load_greencomp_methods(), methods)

    # --- lookups -------------------------------------------------------

    def test_get_all_methods(self):
        self.write_csvs()

        self.assertEqual(
            gc_module.get_all_methods(),
            [
                "Systems Mapping",
                "Case Study Analysis",
                "Futures Literacy Workshop",
                "Policy & Governance Analysis",
            ],
        )

    def test_get_method_description(self):
        self.write_csvs()

        self.assertEqual(
            gc_module.get_method_description("Systems Mapping"),
            "Draw causal loop diagrams",
        )
        self.assertIsNone(gc_module.get_method_description("Unknown method"))

    def test_competency_details_by_competence_name(self):
        self.write_csvs()

        details = gc_module.get_greencomp_competency_details("systems")

        self.assertEqual(
            details,
            {
                "competence_area": "Embracing complexity",
                "competence": "Systems thinking",
                "sample_outcome": "Maps a system",
                "evaluation_methods": "Concept map",
            },
        )

    def test_competency_details_by_area(self):
        self.write_csvs()

        details = gc_module.get_greencomp_competency_details("Envisioning")

        self.assertEqual(details["competence"], "Futures literacy")
        self.assertEqual(details["sample_outcome"], "Imagines scenarios")

    def test_competency_details_not_found(self):
        self.write_csvs()

        self.assertIsNone(gc_module.get_greencomp_competency_details("Nope"))

    def test_methods_for_competencies(self):
        self.write_csvs()

        methods = gc_module.get_suitable_methods_for_competencies(
            ["Systems", "unknown"]
        )

        self.assertEqual(
            [m["name"] for m in methods], ["Systems Mapping", "Case Study Analysis"]
        )
        self.assertEqual(methods[0]["description"], "Draw causal loop diagrams")
        self.assertEqual(methods[0]["best_for"], "Systems thinking")

    def test_methods_for_competencies_no_match(self):
        self.write_csvs()

        self.assertEqual(gc_module.get_suitable_methods_for_competencies(["zzz"]), [])

    def test_methods_for_outcome_text_sorted_by_relevance(self):
        self.write_csvs()

        methods = gc_module.get_suitable_methods_for_outcome_text(
            "Analyze the systems behind sustainability policy"
        )

        names = [m["name"] for m in methods]
        # Systems Mapping: analyze + systems; Case Study: analyze + sustainability
        self.assertEqual(names[:2], ["Case Study Analysis", "Systems Mapping"])
        self.assertEqual(methods[0]["relevance_score"], 2)
        self.assertEqual(methods[1]["relevance_score"], 2)
        self.assertIn("Policy & Governance Analysis", names)
        # methods missing from the CSV are skipped
        self.assertNotIn("Advocacy Campaign", names)

    def test_methods_for_outcome_text_caps_at_five(self):
        rows = "\n".join(
            f"{name},desc {i},src,any"
            for i, name in enumerate(
                [
                    "Case Study Analysis",
                    "Critical Analysis / Deconstruction",
                    "Systems Mapping",
                    "Policy & Governance Analysis",
                    "Futures Literacy Workshop",
                    "Living Lab / Co-creation",
                    "Advocacy Campaign",
                ]
            )
        )
        (self.root / "GreenComp-LearningMethods.csv").write_text(
            "Method_Name,Description,Validation_Source,Best_For_LO_Types\n" + rows,
            encoding="utf-8",
        )

        methods = gc_module.get_suitable_methods_for_outcome_text(
            "analyze and evaluate policy to design the future"
        )

        self.assertEqual(len(methods), 5)

    def test_methods_for_outcome_text_no_keywords(self):
        self.write_csvs()

        self.assertEqual(
            gc_module.get_suitable_methods_for_outcome_text("Learn to swim"), []
        )


if __name__ == "__main__":
    unittest.main()
