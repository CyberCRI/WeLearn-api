// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview
//
// Retry layering (see src/app/tutor/services/b_agents/base.py::BaseAgent.with_retry):
// - this client retries the HTTP call itself (connection errors, timeouts, 429, 5xx;
//   BAML does not filter by status, so other 4xx are retried too) through
//   `retry_policy Exponential`: 1 call + 2 retries, ~0.3s then ~0.45s apart;
// - BaseAgent.with_retry only re-runs the whole function when the LLM output cannot be
//   parsed (BamlValidationError, BamlClientFinishReasonError): 2 attempts.
// Worst case is therefore 2 x 3 = 6 HTTP calls per agent step, each bounded by the
// timeouts below.

client<llm> MistralSmall {
  provider "openai-generic"
  retry_policy Exponential
  options {
    model mistral-small-latest
    api_key env.MISTRAL_API_KEY
    base_url https://api.mistral.ai/v1
    http {
      connect_timeout_ms 10000
      // Syllabus prompts produce long JSON outputs: allow up to 2 minutes per request.
      request_timeout_ms 120000
    }
  }
}




  // load balancer client that alternates between the two Azure clients
  client<llm> CustomFast {
    provider round-robin
    options {
      // This will alternate between the two clients
      strategy [MistralSmall]
    }
  }


// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Constant {
  max_retries 3
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}
