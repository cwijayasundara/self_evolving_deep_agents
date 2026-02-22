# Self-Evolving Deep Agent

A self-improving research agent built on [LangChain deepagents](https://github.com/langchain-ai/deepagents) and [LangGraph](https://github.com/langchain-ai/langgraph). The agent learns new skills from successful runs, optimizes its own prompts from failures, and maintains long-term memory with reflection — all in a closed-loop evolution cycle traced by [LangSmith](https://smith.langchain.com/).

## Architecture

```
                         CLI (run / evolve / prompts / skills / memory)
                                      |
                         +------------v-------------+
                         |   Evolution Orchestrator  |  (LangGraph StateGraph)
                         |   Loops N evolution cycles|
                         +---+------------------+---+
                             |                  |
                  +----------v------+    +------v-----------+
                  | Run Agent Batch |    | Fetch LangSmith  |
                  | (deepagents)    |--->| Traces           |
                  | + Memory Store  |    +------+-----------+
                  +-----------------+           |
                         ^          +-----------v-----------------+
                         |          |      Trajectory Analyzer    |
                  +------+----+     |  Task Comp | Effic. | Qual. |
                  | Memory    |     +------+-------------+--------+
                  | Store     |            |             |
                  | (episodic |  successful|             |failed/partial
                  |  semantic)|            |             |
                  +------^----+  +---------v--+   +------v----------+
                         |       | Skill      |   | Prompt          |
                         |       | Extractor  |   | Optimizer       |
                         |       +-----+------+   +------+----------+
                         |             |                  |
                  +------+----+        v                  v
                  | Reflection|  skills/ dir        prompts/ dir
                  | Engine    |
                  +-----------+
```

**Key loop**: Run agent (with memories) &rarr; Trace via LangSmith &rarr; Analyze trajectories &rarr; Reflect & store memories &rarr; Extract skills from successes + Optimize prompts from failures &rarr; Repeat

### Core Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Deep Agent** | `src/agent/deep_agent.py` | Factory that creates a LangGraph agent using `deepagents.create_deep_agent()` with Tavily search, memory-augmented prompts, and learned skills |
| **Prompt Store** | `src/agent/prompt_store.py` | Versioned prompt persistence as JSON files — tracks score, parent version, and feedback for each prompt iteration |
| **Memory Store** | `src/memory/store.py` | JSON file-backed long-term memory with episodic (past run experiences) and semantic (learned facts/patterns) namespaces |
| **Reflection Engine** | `src/memory/reflection.py` | Post-run LLM analysis that extracts strategy insights, facts, and patterns into episodic + semantic memories |
| **Trajectory Analyzer** | `src/evolution/analyzer.py` | LangGraph workflow that runs three graders sequentially and classifies runs as successful/partial/failed |
| **Graders** | `src/evolution/graders/` | Task completion (LLM-as-judge), efficiency (rule-based: tokens/steps/latency), quality (LLM-as-judge: accuracy/depth/clarity/relevance) |
| **Skill Extractor** | `src/evolution/skill_extractor.py` | Extracts reusable skills from successful trajectories, stored as `SKILL.md` files with YAML frontmatter |
| **Prompt Optimizer** | `src/evolution/prompt_optimizer.py` | Analyzes failure patterns and generates improved system prompts via a metaprompt approach |
| **Orchestrator** | `src/evolution/orchestrator.py` | Top-level LangGraph workflow that loops: run &rarr; trace &rarr; analyze &rarr; reflect &rarr; extract skills &rarr; optimize prompt &rarr; check plateau |
| **Trace Fetcher** | `src/tracing/fetcher.py` | Wraps the LangSmith SDK to fetch and parse run traces into `Trajectory` Pydantic models |

### Evolution Cycle (detailed)

```
tasks.json
    |
    v
[run_batch] ---- Invoke agent on all tasks (with memory-augmented prompts)
    |
    v
[fetch_traces] - Optionally enrich with LangSmith trace data
    |
    v
[analyze] ------ Grade each trajectory through 3 graders, classify result
    |
    v
[reflect] ------ LLM generates reflection; stores episodic + semantic memories
    |
    v
[extract_skills] Extract reusable patterns from successful runs -> skills/
    |
    v
[optimize_prompt] Analyze failures, generate improved prompt -> prompts/v000X.json
    |
    v
[aggregate_metrics] Compute cycle stats, check for plateau
    |
    v
[route] -------- If improving and under max_cycles -> loop back to run_batch
                 Otherwise -> END
```

**Plateau detection**: Evolution stops early if score improvement is less than 5% for 2 consecutive cycles.

**Classification thresholds**: A trajectory is "successful" if at least 2 of 3 graders pass AND the average score is >= 0.75. "Partial" if average >= 0.5. "Failed" otherwise.

### Memory System

The agent maintains three types of memory (following LangGraph's memory taxonomy):

- **Episodic** (`memory/episodic/`): Full run experiences — what task was attempted, what strategy worked or failed, key decisions, and the resulting score. Stored per run as individual JSON files.
- **Semantic** (`memory/semantic/`): Accumulated facts and patterns extracted from reflections — e.g., "Tavily search works better with specific queries", "Medical topics need source verification". Deduplicated over time.
- **Procedural**: Handled by the Prompt Store (versioned system prompts) and Skills system (SKILL.md files).

Before each agent run, relevant memories are retrieved via keyword search and injected into the system prompt as context.

### Skills System

Skills are stored as markdown files with YAML frontmatter in `skills/{skill-id}/SKILL.md`:

```markdown
---
name: Multi-Source Research
description: Research using multiple diverse sources for comprehensive coverage
---

## When to Use
Use when the task requires comprehensive research across multiple domains.

## Steps
1. Search multiple databases with varied queries
2. Cross-reference findings across sources
3. Synthesize into a structured report
```

Skills use **progressive disclosure** — only the name and description are loaded at startup; the full body is read on demand when a task matches the skill description.

## Project Structure

```
self_evolving_deep_agents/
├── pyproject.toml                     # Dependencies, linting, test config
├── .env.example                       # Required environment variables
├── Makefile                           # Dev commands
├── src/
│   ├── __main__.py                    # Entry point (python -m src)
│   ├── config/settings.py             # Pydantic BaseSettings (env-driven)
│   ├── agent/
│   │   ├── deep_agent.py              # create_agent() factory
│   │   ├── prompts.py                 # All prompt templates
│   │   └── prompt_store.py            # Versioned prompt persistence
│   ├── tools/search.py                # Tavily search tool factory
│   ├── memory/
│   │   ├── store.py                   # Long-term memory store (JSON-backed)
│   │   └── reflection.py              # Post-run reflection engine
│   ├── tracing/
│   │   ├── fetcher.py                 # LangSmith trace fetcher
│   │   └── trajectory.py              # Pydantic trajectory data models
│   ├── evolution/
│   │   ├── state.py                   # LangGraph state schemas (TypedDict)
│   │   ├── graders/
│   │   │   ├── task_completion.py     # LLM-as-judge grader
│   │   │   ├── efficiency.py          # Rule-based grader
│   │   │   └── quality.py             # LLM-as-judge grader
│   │   ├── analyzer.py                # Trajectory analyzer (LangGraph)
│   │   ├── skill_extractor.py         # Extract skills from successes
│   │   ├── prompt_optimizer.py        # Metaprompt optimization
│   │   └── orchestrator.py            # Evolution loop (LangGraph)
│   ├── skills/manager.py              # SKILL.md CRUD
│   └── cli/commands.py                # CLI commands
├── skills/                            # Learned skills (SKILL.md files)
├── prompts/                           # Versioned prompt history (JSON)
├── memory/                            # Persistent memory store
│   ├── episodic/                      # Past run experiences
│   └── semantic/                      # Learned facts and patterns
├── tasks/research_tasks.json          # Sample task set
└── tests/unit/                        # 82 unit tests
```

## Setup

### Prerequisites

- Python 3.12+
- API keys for OpenAI, Tavily, and LangChain/LangSmith

### Installation

```bash
# Clone and navigate to the project
cd self_evolving_deep_agents

# Install the package with dev dependencies
pip install -e ".[dev]"

# Copy and fill in your API keys
cp .env.example .env
# Edit .env with your actual keys
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | For OpenAI | — | OpenAI API key |
| `TAVILY_API_KEY` | For search | — | Tavily search API key |
| `LANGCHAIN_API_KEY` | For tracing | — | LangSmith API key |
| `OLLAMA_API_KEY` | For Ollama cloud | — | Ollama cloud API key (triggers cloud routing) |
| `LANGCHAIN_TRACING_V2` | No | `true` | Enable LangSmith tracing |
| `LANGSMITH_PROJECT` | No | `self-evolving-agent` | LangSmith project name |
| `MODEL` | No | `gpt-4o` | Model name (e.g. `gpt-4o`, `qwen3.5:397b`) |
| `MODEL_PROVIDER` | No | `openai` | Provider: `openai`, `ollama`, `anthropic`, `groq`, etc. |
| `MODEL_BASE_URL` | No | — | Custom base URL (e.g. `http://localhost:11434` for local Ollama) |
| `MAX_EVOLUTION_CYCLES` | No | `5` | Max cycles for evolution loop |
| `BATCH_SIZE` | No | `3` | Tasks per evolution batch |

## Usage

### Run a single research task

```bash
python -m src run "What are the latest advances in quantum computing?"
```

The agent will:
1. Retrieve relevant memories from past runs
2. Execute the research task with Tavily search
3. Produce a structured markdown report
4. Traces appear in your LangSmith dashboard

### Run the evolution loop

```bash
python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 5
```

Each cycle:
1. Runs the agent on all tasks (with memory-augmented prompts)
2. Fetches traces from LangSmith
3. Grades each trajectory (task completion, efficiency, quality)
4. Reflects on results and stores episodic + semantic memories
5. Extracts skills from successful runs into `skills/`
6. Optimizes the system prompt based on failure patterns into `prompts/`
7. Checks for plateau — stops early if not improving

Output shows per-cycle metrics:
```
Cycle 0: score=0.650, skills=0, prompt=v1, memories=9
Cycle 1: score=0.720, skills=1, prompt=v2, memories=18
Cycle 2: score=0.785, skills=2, prompt=v2, memories=27
```

### Inspect prompt history

```bash
python -m src prompts
```

Shows all prompt versions with scores, parent lineage, and feedback summaries.

### List learned skills

```bash
python -m src skills
```

Shows all skills extracted from successful runs.

### Browse memories

```bash
python -m src memory
```

Shows counts and recent entries from episodic and semantic memory stores.

### Makefile shortcuts

```bash
make run TASK="Research nuclear fusion breakthroughs"
make evolve CYCLES=3
make prompts
make skills
make memory
```

## Development

```bash
# Install dev dependencies
make dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Lint
make lint

# Format
make format

# Type check
make typecheck
```

### Test Suite

97 unit tests covering all components:

```bash
python -m pytest tests/ -v
```

Tests are fully isolated from the real `.env` file and use mocked LLM responses for deterministic results.

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [deepagents](https://pypi.org/project/deepagents/) | >= 0.4.3 | Core deep agent framework (`create_deep_agent`) |
| [langchain-openai](https://pypi.org/project/langchain-openai/) | >= 1.1.10 | OpenAI model integration |
| [langchain-tavily](https://pypi.org/project/langchain-tavily/) | >= 0.2.17 | Tavily web search tool |
| [langgraph](https://pypi.org/project/langgraph/) | >= 1.0.9 | StateGraph workflows for analyzer and orchestrator |
| [langsmith](https://pypi.org/project/langsmith/) | >= 0.7.6 | Tracing SDK and observability |
| [pydantic-settings](https://pypi.org/project/pydantic-settings/) | >= 2.13.1 | Environment-based configuration |

## System Design — Self-Evolution Mechanics

This section explains how the three self-improvement subsystems work together: **dynamic prompt optimization**, **skill learning**, and **reflective memory**. Each subsystem creates a feedback loop that feeds into the next evolution cycle.

### How It All Connects

```
┌──────────────────────────────────────────────────────────────────────┐
│                       EVOLUTION CYCLE N                              │
│                                                                      │
│  ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌───────────────┐  │
│  │ Prompt   │    │ Memory    │    │ Skills   │    │   Agent       │  │
│  │ Store    │───>│ Store     │───>│ Store    │───>│   Execution   │  │
│  │ (v3)     │    │ (search)  │    │ (load)   │    │   (deepagents)│  │
│  └────▲─────┘    └─────▲─────┘    └────▲─────┘    └──────┬────────┘  │
│       │                │               │                  │          │
│       │                │               │                  ▼          │
│  ┌────┴─────┐    ┌─────┴─────┐    ┌────┴─────┐    ┌──────────────┐  │
│  │ Prompt   │    │Reflection │    │ Skill    │    │  Grading     │  │
│  │Optimizer │◄───│ Engine    │◄───│Extractor │◄───│  Pipeline    │  │
│  │(failures)│    │(all runs) │    │(successes│    │  (3 graders) │  │
│  └──────────┘    └───────────┘    └──────────┘    └──────────────┘  │
│                                                                      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
              CYCLE N+1 uses: improved prompt (v4)
                              + new memories
                              + new skills
```

**Each cycle produces three outputs** that accumulate over time and improve the next cycle:
1. **New prompt version** — addresses specific failure patterns from grading
2. **New memories** — episodic (run records) + semantic (facts/patterns) from reflection
3. **New skills** — reusable strategies extracted from high-scoring runs

---

### 1. Dynamic Prompt Optimization

The system treats the agent's system prompt as a versioned artifact that evolves through failure analysis.

**The closed loop:**

```
                        ┌─────────────────────────────┐
                        │  Current Prompt (v3)        │
                        │  "You are a research agent  │
                        │   ... {memory_context}"     │
                        └──────────┬──────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────────────┐
                        │  Agent runs 3 tasks          │
                        │  with this prompt             │
                        └──────────┬───────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────────────┐
                        │  3 Graders evaluate each run │
                        │  ┌─────────────────────────┐ │
                        │  │ Task Completion (LLM)   │ │
                        │  │ Efficiency (rules)      │ │
                        │  │ Quality (LLM)           │ │
                        │  └─────────────────────────┘ │
                        │  → classify: success/partial │
                        └──────────┬───────────────────┘
                                   │
                     ┌─────────────┴────────────┐
                     │                          │
              failed/partial               successful
                     │                          │
                     ▼                          ▼
          ┌───────────────────┐      (used for skill
          │ Failure Analysis  │       extraction)
          │                   │
          │ Extract common    │
          │ issues from all   │
          │ grader reasoning  │
          │ (top 10 patterns) │
          └────────┬──────────┘
                   │
                   ▼
          ┌───────────────────────────────────────────┐
          │  Metaprompt Invocation                    │
          │                                           │
          │  "You are a prompt engineer.              │
          │   Here is the current prompt: {v3}        │
          │   Current average score: 0.690            │
          │   Failure patterns:                       │
          │   - Missing source citations              │
          │   - Unclear report structure              │
          │   - Unverified numeric claims             │
          │                                           │
          │   Generate an improved prompt that        │
          │   addresses these specific issues."       │
          └────────┬──────────────────────────────────┘
                   │
                   ▼
          ┌───────────────────┐
          │  Save as v4       │
          │  prompts/v0004.json│
          │  parent: v3       │
          │  feedback: [...]  │
          └───────────────────┘
                   │
                   ▼
          Next cycle uses v4 automatically
```

**How the agent picks up the new prompt:** Each call to `create_agent()` calls `prompt_store.get_current_prompt()`, which returns the highest-scoring version (or the latest if no scores are set yet). No manual intervention needed — the next cycle automatically uses the improved prompt.

**Prompt version file** (`prompts/v0003.json`):
```json
{
  "version": 3,
  "prompt": "You are an advanced research agent...",
  "score": null,
  "timestamp": "2026-02-22T11:11:43Z",
  "parent_version": 2,
  "feedback_summary": [
    "task_completion: Missing source citations",
    "quality: Output lacks verifiable claims",
    "task_completion: Report structure unclear"
  ]
}
```

**Convergence control:** Evolution stops when score improvement drops below 5% for 2 consecutive cycles (plateau detection), preventing infinite looping on diminishing returns.

---

### 2. Skill Learning and Reuse

Skills are reusable strategies extracted from high-scoring agent runs. They represent **procedural memory** — "how to do things well."

**Extraction flow:**

```
┌─────────────────────────────────────────────────────────┐
│  Trajectory Analysis Results (one cycle)                │
│                                                         │
│  Run 1: score=0.750 (partial) ──► ELIGIBLE (>= 0.70)  │
│  Run 2: score=0.687 (partial) ──► skip (< 0.70)       │
│  Run 3: score=0.633 (partial) ──► skip (< 0.70)       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  LLM Skill Extraction (for Run 1)                      │
│                                                         │
│  Input:                                                 │
│    task: "Research quantum computing advances"          │
│    tool_calls: [search("quantum error correction"),     │
│                 search("topological qubits 2025"), ...] │
│    output: "# Quantum Computing Report..." (3000 chars) │
│    score: 0.750                                         │
│                                                         │
│  LLM identifies the reusable pattern:                   │
│    name: "structured-research-synthesis"                │
│    description: "Multi-phase research with synthesis"   │
│    content: step-by-step instructions                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Deduplication Check                                    │
│  Is "structured-research-synthesis" similar to          │
│  any existing skill? (checks shared word overlap)       │
│  → No duplicates found                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Save to skills/structured-research-synthesis/SKILL.md  │
│                                                         │
│  ---                                                    │
│  name: structured-research-synthesis                    │
│  description: Multi-phase research with synthesis       │
│  ---                                                    │
│                                                         │
│  ## Steps                                               │
│  1. Break topic into sub-questions                      │
│  2. Search each sub-question with specific queries      │
│  3. Cross-reference findings across sources             │
│  4. Synthesize into structured report with citations    │
└─────────────────────────────────────────────────────────┘
```

**How skills are reused in the next run:**

Skills are passed to the `deepagents` library via the `skills` parameter in `create_deep_agent()`:

```python
# src/agent/deep_agent.py
return create_deep_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt,
    name=AGENT_NAME,
    skills=[str(settings.skills_path)],  # ← learned skills directory
)
```

The deepagents `SkillsMiddleware` handles the rest:
1. **Discovery** — scans the `skills/` directory, loads only YAML frontmatter (name + description)
2. **Matching** — when a task arrives, the middleware identifies which skills are relevant based on description similarity
3. **Injection** — relevant skills' full body content is loaded on demand and injected into the agent's context as a "Skills System" section
4. **Progressive disclosure** — only skill descriptions load at startup; full SKILL.md content is read only when the agent encounters a matching task

This means a skill learned in Cycle 0 is automatically available to the agent in Cycle 1 and all subsequent runs.

---

### 3. Reflective Memory System

The memory system gives the agent **long-term learning** across runs. After every execution, a reflection engine analyzes what happened and stores two types of memories.

**Reflection and storage flow:**

```
┌───────────────────────────────────────────────────────┐
│  After each agent run, the Reflection Engine runs:    │
│                                                       │
│  Input to LLM:                                        │
│    task: "Research quantum computing"                 │
│    output: "# Report..." (truncated)                  │
│    tool_calls: [{search: "quantum"}, ...]             │
│    grader_results: {avg: 0.75, tc: FAIL, q: FAIL}    │
│                                                       │
│  LLM generates structured reflection:                 │
│    strategy: "Used iterative search refinement"       │
│    worked_well: ["Clear organization", ...]           │
│    improvements: ["Add source citations", ...]        │
│    facts: ["Bosonic codes are a key direction", ...]  │
│    patterns: ["ANTI-PATTERN: numeric claims without   │
│               citations", ...]                        │
└──────────────┬─────────────────────┬──────────────────┘
               │                     │
               ▼                     ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│  EPISODIC MEMORY     │  │  SEMANTIC MEMORY              │
│  memory/episodic/    │  │  memory/semantic/             │
│                      │  │                               │
│  One file per run:   │  │  Many small files:            │
│  {                   │  │                               │
│    run_id, task,     │  │  fact-7c0c2863.json:          │
│    summary,          │  │  { type: "fact",              │
│    worked_well,      │  │    content: "Bosonic codes    │
│    improvements,     │  │    are a key QC direction" }  │
│    score,            │  │                               │
│    output_preview    │  │  pattern-403f3227.json:       │
│  }                   │  │  { type: "pattern",           │
│                      │  │    content: "ANTI-PATTERN:    │
│  Records WHAT        │  │    numeric claims without     │
│  happened            │  │    citations" }               │
│                      │  │                               │
│                      │  │  Records WHAT WAS             │
│                      │  │  LEARNED                      │
└──────────────────────┘  └──────────────────────────────┘
```

**How memories are used in the next run:**

Before each agent execution, `build_memory_context()` searches both memory stores by keyword matching against the current task:

```
Task: "Research nuclear fusion energy"
                │
                ▼
┌──────────────────────────────────────────────────────┐
│  Keyword search across all memories                  │
│                                                      │
│  Episodic hits (top 3):                             │
│  - "Used iterative search for energy research"       │
│  - "Multi-source approach worked well for physics"   │
│                                                      │
│  Semantic hits (top 5):                             │
│  - "Specific queries yield better Tavily results"    │
│  - "ANTI-PATTERN: claims without citations"          │
│  - "Energy topics need recent sources (2024+)"       │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Injected into system prompt via {memory_context}:   │
│                                                      │
│  ## Relevant Past Experience                         │
│  ### Previous Runs                                   │
│  - Used iterative search for energy research         │
│  - Multi-source approach worked well for physics     │
│  ### Learned Facts & Patterns                        │
│  - Specific queries yield better Tavily results      │
│  - ANTI-PATTERN: claims without citations            │
│  - Energy topics need recent sources (2024+)         │
└──────────────────────────────────────────────────────┘
```

The agent sees this context at the end of its system prompt, allowing it to avoid past mistakes and apply learned strategies.

---

### Putting It All Together — A Full Evolution Run

Here's what happens when you run `python -m src evolve --max-cycles 3`:

| Phase | Cycle 0 | Cycle 1 | Cycle 2 |
|-------|---------|---------|---------|
| **Prompt** | v1 (default) | v2 (addresses Cycle 0 failures) | v3 (addresses Cycle 1 failures) |
| **Memories** | None (first run) | 74 memories from Cycle 0 | 111 memories from Cycles 0-1 |
| **Skills** | None | 1 skill from Cycle 0 success | 1-2 skills accumulated |
| **Agent context** | Base prompt only | Prompt v2 + relevant memories + skills | Prompt v3 + more memories + more skills |
| **Avg score** | 0.690 | 0.720 (improving) | 0.785 (improving) |
| **What improved** | — | Citations added (from failure feedback) | Structure improved (from learned patterns) |

**Real output from a 2-cycle run:**
```
============================================================
  CYCLE 0  |  Running 3 tasks  |  prompt v2
============================================================
  [b4415e40] quantum computing  -> PARTIAL (avg=0.750)
  [5f0582f9] nuclear fusion     -> PARTIAL (avg=0.687)
  [a9006328] LLM safety         -> PARTIAL (avg=0.633)
  Analysis summary: 0 successful, 3 partial, 0 failed
  Reflection stored: 4 facts, 4 patterns for run b4415e40
  NEW SKILL: structured-research-synthesis
  Prompt upgraded: v2 -> v3 (4507 chars)
------------------------------------------------------------
  CYCLE 0 COMPLETE  |  avg_score=0.690  |  skills=1  |  memories=74
------------------------------------------------------------

============================================================
  CYCLE 1  |  Running 3 tasks  |  prompt v3
============================================================
  ...grading with improved prompt and accumulated memories...
  Prompt upgraded: v3 -> v4 (4603 chars)
------------------------------------------------------------
  CYCLE 1 COMPLETE  |  avg_score=0.621  delta=-0.069  |  memories=111
------------------------------------------------------------
```

### Multi-Provider Support

The system supports 20+ LLM providers via `init_chat_model()`. Ollama cloud models are automatically routed through the OpenAI-compatible endpoint at `https://ollama.com/v1`.

```bash
# .env — switch between providers by commenting/uncommenting:

# Ollama cloud
MODEL=qwen3.5:397b
MODEL_PROVIDER=ollama
OLLAMA_API_KEY=...

# OpenAI
# MODEL=gpt-4o
# MODEL_PROVIDER=openai
```
