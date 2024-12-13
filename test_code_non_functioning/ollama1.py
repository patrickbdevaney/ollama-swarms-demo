import sys  # Add this import to avoid the NameError

from loguru import logger
import subprocess
from typing import Optional
from datetime import datetime
from swarms import Agent, AgentRearrange, create_file_in_folder
import ollama

# Configure logging
logger.remove()  # Remove default logger configuration

# Log to console
logger.add(sys.stdout, level="INFO")

# Log to file (e.g., 'diagnosis_log.log')
logger.add("diagnosis_log.log", level="INFO", rotation="500 MB", retention="10 days", compression="zip")

# Ensure ollama-python is installed
try:
    import ollama
except ImportError:
    logger.error("Failed to import ollama. Installing...")
    subprocess.run(["pip", "install", "ollama-python"], check=True)
    import ollama

# Test connection to Ollama and send a basic message
def test_connection_and_generate_sample(host="http://127.0.0.1:11434"):
    logger.info(f"Attempting to connect to Ollama at {host}...")
    
    try:
        # Test connection by listing models
        models = ollama.models.list_models(base_url=host)
        if not models:
            logger.error("No models found. Connection might have failed or no models available.")
            return False
        logger.info("Connection to Ollama was successful. Models available.")

        # Test with a basic completion request
        sample_prompt = "What is the capital of France?"
        response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": sample_prompt}])

        if response:
            logger.info(f"Sample completion response: {response}")
            return True
        else:
            logger.error("Sample completion failed: No response from Ollama.")
            return False
            
    except Exception as e:
        logger.error(f"An error occurred while connecting to Ollama or generating a sample response: {e}")
        return False


# Define the Ollama Model for Swarms Agents
class OllamaModel:
    def __init__(self, model_name: str, host: Optional[str] = None, timeout: int = 30, max_tokens: int = 4000):
        self.model_name = model_name
        self.host = host or "http://127.0.0.1:11434/"  # Default to local Ollama server
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> Optional[str]:
        if not prompt:
            logger.error("Prompt cannot be empty.")
            return None

        try:
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return None


# Define Swarms agents for medical diagnosis
chief_medical_officer = Agent(
    agent_name="Chief Medical Officer",
    system_prompt="""
        You are the Chief Medical Officer coordinating a team of medical specialists for viral disease diagnosis.
        Your responsibilities include:
        - Gathering initial patient symptoms and medical history
        - Coordinating with specialists to form differential diagnoses
        - Synthesizing different specialist opinions into a cohesive diagnosis
        - Ensuring all relevant symptoms and test results are considered
        - Making final diagnostic recommendations
        - Suggesting treatment plans based on team input
        - Identifying when additional specialists need to be consulted
        - For each differential diagnosis, provide minimum and maximum lab ranges to meet that diagnosis.
    """,
    llm=OllamaModel(model_name="llama3.1:8b"),
    max_loops=1,
)

virologist = Agent(
    agent_name="Virologist",
    system_prompt="""
        You are a specialist in viral diseases. For each case, provide:
        - Detailed viral symptom analysis
        - Disease progression timeline
        - Risk factors and complications
        - Relevant ICD-10 codes for confirmed viral conditions, suspected viral conditions, associated symptoms, and complications
    """,
    llm=OllamaModel(model_name="llama3.1:8b"),
    max_loops=1,
)

internist = Agent(
    agent_name="Internist",
    system_prompt="""
        You are an Internal Medicine specialist responsible for comprehensive evaluation:
        - Review system-by-system
        - Vital signs analysis
        - Comorbidity evaluation
        - Assign appropriate ICD-10 codes and include hierarchical condition category (HCC) codes where applicable
    """,
    llm=OllamaModel(model_name="llama3.1:8b"),
    max_loops=1,
)

# Define diagnostic flow for medical task
flow = f"{chief_medical_officer.agent_name} -> {virologist.agent_name} -> {internist.agent_name}"

# Create the swarm system for medical diagnosis
diagnosis_system = AgentRearrange(
    name="Medical-coding-diagnosis-swarm",
    description="Comprehensive medical diagnosis and coding system",
    agents=[chief_medical_officer, virologist, internist],
    flow=flow,
    max_loops=1,
    output_type="all",
)

# Generate a report
def generate_report(diagnosis_output: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# Medical Diagnosis and Coding Report
Generated: {timestamp}

## Clinical Summary
{diagnosis_output}

## Recommendations
- Additional documentation needed
- Suggested follow-up
- Coding optimization opportunities
"""
    return report


# Main execution
if __name__ == "__main__":
    # Test Ollama connection first
    if test_connection_and_generate_sample():
        logger.info("Connection to Ollama successful. Running diagnostic agents.")

        # Sample patient case
        patient_case = """
        Patient: 45-year-old White Male
        Lab Results:
        - eGFR: 59 ml/min/1.73 m^2 (non-African American)
        """

        # Run the diagnostic process with Swarms agents
        logger.info(f"Running diagnostic for patient case: {patient_case}")
        diagnosis = diagnosis_system.run(patient_case)
        logger.info(f"Diagnosis generated: {diagnosis}")

        # Generate a report from the diagnosis
        report = generate_report(diagnosis)
        logger.info(f"Generated report: {report}")
    else:
        logger.error("Connection to Ollama failed. Exiting...")
