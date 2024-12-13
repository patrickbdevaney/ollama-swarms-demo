import ollama

# Test call to the model
response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": "Hello, model!"}])

print(response)