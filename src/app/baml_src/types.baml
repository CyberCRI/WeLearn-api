// BAML Type Definitions for WeLearn MAS
// These types match the Pydantic models in models.py

// Enums
enum SessionMode {
  PRESENTIEL @alias("Présentiel")
  DISTANCIEL @alias("Distanciel")
  HYBRIDE @alias("Hybride")
}

// Core Types
class CourseMetadata {
  discipline string
  topic string
  level string
  num_sessions int
  session_duration float
  session_type string
  session_mode SessionMode
  class_size int
  output_language string
  user_description string?
}

class Document {
  text string
  corpus string
  description string
  sdg int[]
  url string
  title string
  relevance_score float?
}

class CourseDescription {
  text string
  word_count int
}

class LearningObjective {
  number int
  text string
  bloom_level string?
}

class LearningObjectives {
  objectives LearningObjective[]
}

class SustainabilityConnection {
  objective_number int
  sdg_themes string[]
  connection_explanation string
  key_resources Document[]
}

class SustainabilityIntegration {
  connections SustainabilityConnection[]
  suggested_objectives LearningObjective[]?
  integration_strategy string
  resources_used Document[]
}

class LearningOutcome {
  number int
  text string
  related_objectives int[]
  assessment_method string
}

class LearningOutcomes {
  outcomes LearningOutcome[]
}

class CompetencyMapping {
  outcome_number int
  greencomp_competencies string[]
  rationale string
}

class CompetencyMappings {
  mappings CompetencyMapping[]
}

class ValidationReport {
  passed bool
  issues string[]
  suggestions string[]
  severity string
}

class SessionPlan {
  session_number int
  objectives int[]
  type string
  mode SessionMode
  duration float
  class_size int
}

class ActivityTemplate {
  id string
  name string
  description string
  compatible_types string[]
  compatible_modes SessionMode[]
  min_duration float
  max_duration float
  min_size int
  max_size int
  estimated_duration float
}

class ActivityGuide {
  activity_name string
  description string
  learning_objectives int[]
  steps string[]
  teacher_role string
  student_role string
  resources_needed string[]
  timing_breakdown map<string, float>
  evaluation_method string
  sustainability_integration string
}

class ActivityRecommendations {
  session_number int
  recommended_activities ActivityTemplate[]
  rationale string
}
