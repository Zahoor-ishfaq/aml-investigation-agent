from aml_agent.agent.groq_client import chat

resp = chat(messages=[{"role": "user", "content": "Say 'AML pipeline ready' and nothing else."}])
print(resp.choices[0].message.content)