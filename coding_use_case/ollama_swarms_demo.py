import json
import requests
from datetime import datetime
from swarms import Agent, AgentRearrange
from swarm_models import OllamaModel

class OllamaModel:
    def __init__(self, model_name="qwen2.5-coder:32b-instruct-q4_K_M"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/generate"

    def __call__(self, prompt):
        try:
            # Stream the response and collect it
            full_response = ""
            for line in self._stream_ollama_api_call(prompt):
                try:
                    json_line = json.loads(line)
                    if 'response' in json_line:
                        full_response += json_line['response']
                except json.JSONDecodeError:
                    continue
            return full_response.strip()
        except Exception as e:
            print(f"API call error: {e}")
            return f"Error in API call: {str(e)}"

    def _stream_ollama_api_call(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True
        }
        try:
            response = requests.post(self.base_url, json=payload, stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield line.decode('utf-8')
        except requests.RequestException as e:
            print(f"Request error: {e}")
            yield json.dumps({"error": str(e)})

# Create model instance
model = OllamaModel(model_name="qwen2.5-coder:32b-instruct-q4_K_M")

# Define agents
code_writer = Agent(
    agent_name="Code Writer",
    system_prompt="""You are a skilled code writer... (omitted for brevity)""",
    llm=model,
    max_loops=1,
)

code_reviser_1 = Agent(
    agent_name="Code Reviser 1",
    system_prompt="""You are an expert code reviewer... (omitted for brevity)""",
    llm=model,
    max_loops=1,
)

code_reviser_2 = Agent(
    agent_name="Code Reviser 2",
    system_prompt="""You are a senior code reviewer... (omitted for brevity)""",
    llm=model,
    max_loops=1,
)

# Define flow and swarm system
agents = [code_writer, code_reviser_1, code_reviser_2]
flow = f"""{code_writer.agent_name} -> {code_reviser_1.agent_name} -> {code_reviser_2.agent_name}"""

code_generation_system = AgentRearrange(
    name="CodeGenerationSwarm",
    description="Swarm system for generating and reviewing Python scripts across various tasks",
    agents=agents,
    flow=flow,
    max_loops=1,
    output_type="all",
)

def generate_code_for_task(task_description: str) -> tuple:
    """
    Generate the Python script for the specified task and collect outputs.
    """
    case_info = f"Timestamp: {datetime.now()}\nTask Description: {task_description}"
    
    # Generate code
    draft_code = code_writer(case_info)
    
    # Revision 1
    revision_1_code = code_reviser_1(draft_code)
    
    # Revision 2
    revision_2_code = code_reviser_2(revision_1_code)
    
    return draft_code, revision_1_code, revision_2_code

def save_delimited_outputs(draft_code: str, revision_1_code: str, revision_2_code: str):
    """
    Save the outputs to three separate files with appropriate delimiters.
    """
    # Save draft code
    with open("draft.py", "w") as draft_file:
        draft_file.write("# Code Writer Output\n")
        draft_file.write("#" * 80 + "\n")
        draft_file.write(draft_code)

    # Save revision 1 code
    with open("revision1.py", "w") as revision1_file:
        revision1_file.write("# Code Reviser 1 Output\n")
        revision1_file.write("#" * 80 + "\n")
        revision1_file.write(revision_1_code)

    # Save revision 2 code
    with open("revision2.py", "w") as revision2_file:
        revision2_file.write("# Code Reviser 2 Output\n")
        revision2_file.write("#" * 80 + "\n")
        revision2_file.write(revision_2_code)

    print("Files generated successfully!")

if __name__ == "__main__":
    # Example task description
    task_description = """
    Task: Generate a Python script for a machine learning task.

    The script should:
    1. Define a neural network model using a deep learning library.
    2. Implement the training and validation loop.
    3. Use an appropriate optimizer and loss function.
    4. Load and preprocess datasets.
    5. Implement mechanisms to avoid overfitting.
    6. Log training metrics and save the trained model.
    """

    # Generate code and revisions
    draft_code, revision_1_code, revision_2_code = generate_code_for_task(task_description)

    # Save outputs with delimiters
    save_delimited_outputs(draft_code, revision_1_code, revision_2_code)
