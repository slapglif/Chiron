# agents.md - Root Orchestrator & Swarm Protocol

## Hermeneutic Circle Framework

This agent team operates using **hermeneutic circle thinking**: understanding flows bidirectionally between the whole system and its parts. Each agent interprets its domain in context of the whole pipeline, and the whole pipeline's meaning emerges from the interplay of its parts.

### Circle of Understanding

```
                    ┌─────────────────────┐
                    │   WHOLE (Pipeline)   │
                    │  Semantic Folding    │
                    │  Text → SDR → Model │
                    └──────────┬──────────┘
                               │
              Interpretation flows down
              Understanding flows up
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        v          v           v           v          v
   ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Preproc  │ │ Layers │ │  Eval  │ │ Utils  │ │ Tests  │
   │ Agent   │ │ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │
   └─────────┘ └───┬────┘ └────────┘ └────────┘ └────────┘
                    │
           ┌───────┼───────┐
           v       v       v
        ┌─────┐ ┌─────┐ ┌─────┐
        │ HTM │ │ SDR │ │ SNN │
        │Agent│ │Agent│ │Agent│
        └─────┘ └─────┘ └─────┘
```

Each iteration of understanding:
1. **Pre-understanding**: Agent reads its local `agents.md` for domain context
2. **Engagement**: Agent examines the specific code and task at hand
3. **Interpretation**: Agent acts within its domain expertise
4. **Revised understanding**: Insights feed back up to parent agents and across to siblings
5. **Fusion of horizons**: Cross-domain changes reconcile with the whole system

## Swarm Mode Protocol

### Agent Hierarchy

| Level | Agent | Domain | Files Owned |
|-------|-------|--------|-------------|
| L0 | **Root Orchestrator** | Full pipeline coordination | `CLAUDE.md`, `agents.md`, `.gitignore` |
| L1 | **Pipeline Agent** | Core module orchestration | `chiron/main.py`, `chiron/train.py`, `chiron/pipeline.py`, `chiron/config.json` |
| L2 | **Preprocessing Agent** | Text processing & embeddings | `chiron/preprocessing/*` |
| L2 | **Layers Orchestrator** | Neural architecture coordination | `chiron/layers/__init__.py` |
| L3 | **HTM Agent** | Hierarchical Temporal Memory | `chiron/layers/htm/*` |
| L3 | **SDR Agent** | Sparse Distributed Representations | `chiron/layers/sdr/*` |
| L3 | **SNN Agent** | Spiking Neural Networks & Graph Attention | `chiron/layers/snn/*` |
| L2 | **Evaluation Agent** | Metrics & downstream tasks | `chiron/evaluation/*` |
| L2 | **Utilities Agent** | Config, data, caching | `chiron/utils/*` |
| L2 | **Testing Agent** | Test coverage & validation | `chiron/tests/*` |

### Delegation Rules

1. **Top-down delegation**: Tasks flow from Root → L1 → L2 → L3 based on domain ownership
2. **Bottom-up escalation**: When a change in one domain affects another, escalate to the nearest common ancestor
3. **Lateral consultation**: Sibling agents consult each other for interface contracts
   - Preprocessing ↔ Layers: embedding format, tensor shapes
   - Layers ↔ Evaluation: model output format, metric inputs
   - Utils ↔ All: config keys, data format, caching API

### Swarm Execution Model

```
Phase 1: INTERPRET (Hermeneutic Pre-understanding)
  - Root agent reads CLAUDE.md for project context
  - Task is decomposed into domain-specific subtasks
  - Each relevant agent reads its local agents.md

Phase 2: DELEGATE (Swarm Dispatch)
  - Subtasks dispatched to domain agents in parallel where independent
  - Sequential dispatch where data dependencies exist:
    Preprocessing → SDR → Adjacency → SNN/GAT/HTM → Evaluation

Phase 3: EXECUTE (Domain Expert Action)
  - Each agent works within its owned files
  - Cross-cutting changes require escalation
  - Agents validate against invariants in their agents.md

Phase 4: INTEGRATE (Hermeneutic Fusion)
  - Changes reconciled at the orchestrator level
  - Interface contracts verified across agent boundaries
  - Whole-system understanding updated
```

### Data Flow Contract

The pipeline has a strict data flow. Agents must preserve these interfaces:

```
TextPreprocessor.preprocess() -> List[str]
    Input: List[List[Dict[str, Any]]]  (conversations)
    Output: List[str]  (preprocessed text tokens)

Word2VecEmbedding.generate_embeddings() -> List[np.ndarray]
    Input: List[str]  (preprocessed conversations)
    Output: List[np.ndarray]  (300-dim float vectors)

SDRGenerator.generate_sdr_embeddings() -> np.ndarray
    Input: List[np.ndarray]  (word embeddings)
    Output: np.ndarray  (binary SDR matrix, shape [N, projection_dims])

compute_and_save_adjacency_matrix() -> (csr_matrix, int)
    Input: torch.Tensor  (SDR embeddings)
    Output: (scipy.sparse.csr_matrix, num_embeddings)

SNNModel.forward() -> torch.Tensor
    Input: input_ids, attention_mask, adjacency_matrix, node_indices
    Output: torch.Tensor  (batch_size, sdr_dimensions)
```

### Cross-Cutting Concerns

These concerns span multiple agents and require coordination:

| Concern | Agents Involved | Coordination Point |
|---------|----------------|-------------------|
| Tensor shape changes | Preprocessing, Layers, Evaluation | Pipeline Agent (L1) |
| Config key additions | Utils, all consumers | Root Orchestrator (L0) |
| New evaluation metrics | Evaluation, Training | Pipeline Agent (L1) |
| GPU/CPU device handling | All layer agents | Layers Orchestrator (L2) |
| Logging conventions | All agents | Root Orchestrator (L0) |

### Agent Communication Patterns

**Handoff Pattern**: When preprocessed data flows to layers
```
Preprocessing Agent produces -> validates shape/type -> Layers Agent consumes
```

**Broadcast Pattern**: When config changes affect multiple agents
```
Utils Agent changes config -> Root Orchestrator notifies -> All affected agents adapt
```

**Convergence Pattern**: When evaluation needs outputs from multiple layers
```
SNN Agent output + HTM Agent output + GAT Agent output -> Evaluation Agent aggregates
```

## Deep Init Strategy

Each subdirectory's `agents.md` follows this structure:
1. **Role**: What this agent is responsible for
2. **Hermeneutic Context**: How this part relates to the whole
3. **Domain Expertise**: Technical knowledge required
4. **Owned Files**: Files this agent manages
5. **Invariants**: Rules that must never be broken
6. **Interfaces**: Input/output contracts with other agents
7. **Delegation**: When to escalate or consult
