// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

// Example Azure OpenAI client (uncomment to use)
client<llm> CustomAzureFrance {
  provider azure-openai
  options {
    model env.LLM_MODEL_NAME
    api_key env.AZURE_LLM_FRANCE_API_KEY
    base_url env.AZURE_LLM_FRANCE_BASE_URL
    api-version "2024-05-01-preview"
  }
}

client<llm> CustomAzureSweden {
  provider azure-openai
  options {
    model env.LLM_MODEL_NAME
    api_key env.AZURE_LLM_SWEDEN_API_KEY
    base_url env.AZURE_LLM_SWEDEN_BASE_URL
    api-version "2024-05-01-preview"
  }
}


client<llm> ClientLoadBalancer {
  provider round-robin
  options {
    strategy [
      CustomAzureFrance
      CustomAzureSweden
    ]
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