This README provides a comprehensive guide on setting up and running sequential swarms using Ollama. Here's a breakdown of the key points:

**Introduction**

* The repository demonstrates two sequential agent swarms: Code Review and Medical Team.
* Both swarms use Ollama's API and a streaming approach for efficient communication between agents.

**Prerequisites**

* Python 3.7+ must be installed.
* Ollama must be running locally.
* The Requests library must be installed using `pip install requests`.

**Setting Up Ollama**

1. Download and install Ollama from the official website.
2. Run Ollama locally using `ollama start`.
3. Verify the server by visiting `http://localhost:11434` in your browser.
4. Test Ollama using a simple request to generate text from a prompt.

**Running the Sequential Swarm**

* **Code Review Sequential Swarm**:
	+ Clone the repository and modify the task description.
	+ Run the swarm using `python run_code_review_swarm.py`.
	+ The generated code will be saved to three separate files: `draft.py`, `revision1.py`, and `revision2.py`.
* **Medical Team Sequential Swarm**:
	+ Modify the task description to reflect the medical task.
	+ Run the swarm using `python run_medical_team_swarm.py`.


**Configuration Details**

* Model Name: Can be modified to use any available model.
* Prompt: Can be modified to define the task for code generation or medical diagnosis.
* Agent Flow: Can be modified to change the order in which agents process the task.

**Customizing Agents**

* Agents can be modified by editing their respective system prompts.
* Each agent can be tweaked to perform a specific role or handle different tasks.

**License**

* The repository is licensed under the MIT License.

To get started, follow these steps:

1. Install Python 3.7+ and the Requests library.
2. Download and install Ollama.
3. Run Ollama locally and verify the server.
4. Clone the repository and modify the task description for the desired swarm.
5. Run the swarm using the provided Python script.
6. Review the generated code or medical diagnosis results.
