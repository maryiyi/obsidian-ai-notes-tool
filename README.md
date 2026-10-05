# obsidian-ai-notes-tool
An automated, AI-driven pipeline that transforms raw, unstructured audio transcripts (e.g., from Zoom or voice memos) into highly structured, Obsidian-ready Markdown notes. 

## The Problem
University students often record lectures or use transcription tools, but raw transcripts are practically unreadable. They lack punctuation, contain filler words, and bury core concepts under conversational tangents. Manually editing them takes hours.

## The Solution
This tool uses a Python script to pass raw transcripts to a local LLM via LM Studio. Instead of a basic zero-shot query, it utilizes a strict **System Prompt Harness** (few-shot prompting and constraint-based persona adoption) to force the model to:
1. Ignore filler words and off-topic discussions.
2. Extract core concepts, definitions, and formulas.
3. Format the output with strict Markdown, including `[[Wikilinks]]` and correct LaTeX syntax for math.
4. Save the generated `.md` file directly into a local Obsidian Vault.

## Architecture & Tech Stack
- **Language:** Python 3.10+
- **LLM Endpoint:** Local OpenAI-compatible API via [LM Studio](https://lmstudio.ai/) (`http://localhost:1234/v1`).
- **Recommended Model:** `Mistral-Nemo-12B-Instruct` or `Llama-3.1-8B`.
- **Integration:** Direct file I/O to local Obsidian Vaults.

## Agentic Workflow & Prompt Engineering
A key feature of this project is the iterative approach to prompt engineering:
- **Initial Naive Approach:** The model suffered from hallucinations, conversational filler ("Here is your summary!"), and broken LaTeX math blocks.
- **The Refinement (Harness):** Implemented a dedicated `harness.md` system prompt. It acts as a rigid constraint, providing few-shot examples of correct formatting and explicitly forbidding conversational tokens, ensuring the output is 100% Obsidian-compliant.

---

## Quick Start

### Prerequisites
- Python 3.10+
- [LM Studio](https://lmstudio.ai/) installed and running

### 1. Clone the repository & set up the environment
```bash
git clone https://github.com/yourusername/obsidian-ai-notes-tool.git
cd obsidian-ai-notes-tool
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Local LLM Server
1. Open LM Studio.
2. Load your model (e.g., `Mistral-Nemo-12B-Instruct` or `Llama-3.1-8B`).
3. Switch to the **Local Server** tab (`↔️`).
4. Click **Start Server** (default: `http://localhost:1234/v1`).

### 3. Run the Pipeline
Place your raw transcript file in the project directory, then run:
```bash
python main.py --input transcript.txt
```
The formatted Markdown note will be created directly inside your Obsidian Vault.

---

## Configuration

Settings are managed via a `.env` file. Create one by copying the example:

```bash
cp .env.example .env
```

Set your values in `.env`:

```env
# Absolute path to your Obsidian vault target folder
OBSIDIAN_VAULT_PATH="/Users/YourName/Documents/Obsidian/University"

# Local endpoint for LM Studio
API_BASE_URL="http://localhost:1234/v1"

# Placeholder key (required by the OpenAI SDK format, ignored by LM Studio)
API_KEY="lm-studio"
```

### Customizing the Prompt Harness
Formatting rules and output templates are stored in `harness.md`. You can freely adjust:
- Tagging conventions (e.g., `#lecture`, `#definition`)
- Wikilink frequency rules
- LaTeX block preferences

The script dynamically injects `harness.md` as the system instructions for every run.