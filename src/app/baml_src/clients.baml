// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview


// Example Azure OpenAI client (uncomment to use)
client<llm> CustomAzure_sweden {
  provider openai
  options {
    model "gpt-oss-120b"
    api_key env.AZURE_LLM_SWEDEN_API_KEY
    base_url env.AZURE_BAML_LLM_SWEDEN_BASE_URL
  }
}

// client<llm> CustomAzure_uk {
//   provider azure-openai
//   options {
//     model "gpt-5"
//     api_key env.AZURE_LLM_SWEDEN_API_KEY
//     base_url env.AZURE_LLM_UK_BASE_URL
//     api_version "2024-05-01-preview"
//   }
// }

client<llm> CustomAzure_france {
  provider openai
  options {
    model "gpt-oss-120b"
    api_key env.AZURE_LLM_FRANCE_API_KEY
    base_url env.AZURE_BAML_LLM_FRANCE_BASE_URL
  }
}

  // load balancer client that alternates between the two Azure clients
  client<llm> CustomFast {
    provider round-robin
    options {
      // This will alternate between the two clients
      strategy [CustomAzure_france, CustomAzure_sweden]
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