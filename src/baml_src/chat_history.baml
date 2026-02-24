class MyUserMessage {
  role "user" | "assistant"
  content string
}

class AgentAnswer {
  answer string
}

function ChatWithLLM(messages: MyUserMessage[]) -> AgentAnswer {
  client ClientLoadBalancer 
  prompt #"
    Answer the user's questions based on the chat history:
    {% for message in messages %}
      {{ _.role(message.role) }} 
      {{ message.content }}
    {% endfor %}

    Answer:
    {{ ctx.output_format }}
  "#
}

test TestName {
  functions [ChatWithLLM]
  args {
    messages [
      {
        role "user"
        content "Hello! How are you? My name is Vaibhav."
      }
      {
        role "assistant"
        content "Hi!"
      }
    ]
  }
}

