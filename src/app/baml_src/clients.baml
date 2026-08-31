// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> MistralSmall {
  provider "openai-generic"
  options {
    model mistral-small-latest
    api_key env.MISTRAL_API_KEY
    base_url https://api.mistral.ai/v1
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