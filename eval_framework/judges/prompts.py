"""LLM judge prompt templates — batch mode.

Each session: 2 LLM calls (recall + correctness).
Each QA question: 2 LLM calls (answer + evidence).
"""

# ---------------------------------------------------------------------------
# Session Call 1: Recall (Gold -> Snapshot)
# ---------------------------------------------------------------------------
RECALL_BATCH_PROMPT = """You are a **Memory Recall Evaluator**.
Determine how many of the **Expected Memory Points** are covered by the system's **Extracted Memories**.

# Inputs

1. **Extracted Memories** (what the system actually stored):
{memories}

2. **Expected Memory Points** (numbered):
{gold_points}

# Instructions

- Go through each Expected Memory Point and check whether the Extracted Memories contain information that covers it.
- Semantic matching is acceptable; exact wording is NOT required.

# Output

```json
{{
  "covered_count": <int>,
  "covered_indices": [<1-based int indices of covered expected memory points>],
  "total": <int>,
  "reasoning": "Brief summary"
}}
```
"""

# ---------------------------------------------------------------------------
# Session Call 2: Correctness (Snapshot -> Gold)
# ---------------------------------------------------------------------------
CORRECTNESS_BATCH_PROMPT = """You are a **Memory Correctness Evaluator**.
Evaluate whether each memory stored by the system is factually correct, using both the golden memory points and the session dialogue as ground truth.

# Inputs

1. **System Memories** ({num_memories} items, numbered, what the system actually stored):
{memories}

2. **Golden Memory Points** (tagged [normal] for new facts or [update] for the updated version of a prior fact):
{gold_points}

3. **Session Dialogue** (the actual conversation that took place):
{dialogue}

# Instructions

For **EVERY SINGLE** System Memory (from [1] to [{num_memories}]), judge the **overall meaning** against the dialogue and golden points, then classify:

- **correct**: The overall meaning of this memory is right — it captures the gist of what happened in the dialogue. Minor extra details or slight imprecision are fine as long as the core message is accurate and not misleading.
- **hallucination**: The overall meaning is partially right, but the memory **also** states something that clearly contradicts the dialogue (e.g. says an action succeeded when it failed, or attributes an action to the wrong object). The contradiction must be about a concrete fact, not just a missing detail.
- **error**: The memory is fundamentally wrong — its core claim contradicts the dialogue, or it describes events/facts that never occurred in the dialogue at all.

**CRITICAL RULES**:
1. You MUST return EXACTLY {num_memories} items in the "results" array — one for each System Memory, in order.
2. Use the dialogue as the primary source of truth for what actually happened.
3. Use the golden points to understand what the key facts are.
4. Be lenient on details: if the gist is correct, label it **correct** even if some minor details are missing or imprecise.

# Output

```json
{{
  "results": [
    {{"id": 1, "label": "correct|hallucination|error"}},
    {{"id": 2, "label": "correct|hallucination|error"}}
  ],
  "reasoning": "Brief justification"
}}
```
"""

# ---------------------------------------------------------------------------
# Session: Update handling (per update gold point)
# ---------------------------------------------------------------------------
UPDATE_EVAL_PROMPT = """You are a **Memory Update Evaluator**.
Determine how a memory system handled an information update.

# Inputs

1. **System Memories** (what the system currently stores after this session):
{memories}

2. **Updated Fact** (the NEW correct information):
{new_content}

3. **Outdated Fact(s)** (the OLD information that should have been replaced):
{old_contents}

# Instructions

Check the System Memories and classify the update handling as one of:

- **updated**: The system stores ONLY the new/updated information. The outdated fact is no longer present. This is the ideal outcome.
- **both**: The system stores BOTH the new and the old information. The update was partially handled — the new fact was added but the old was not removed.
- **outdated**: The system stores ONLY the old/outdated information. The update was missed entirely — the new fact is absent.

Use semantic matching, not exact wording.

# Output

```json
{{
  "label": "updated|both|outdated",
  "reasoning": "Brief justification"
}}
```
"""

# ---------------------------------------------------------------------------
# Session: Interference rejection (per interference gold point)
# ---------------------------------------------------------------------------
INTERFERENCE_EVAL_PROMPT = """You are a **Memory Interference Evaluator**.
Determine whether a memory system incorrectly stored information that should have been ignored.

# Inputs

1. **System Memories** (what the system currently stores after this session):
{memories}

2. **Interference Content** (information that should NOT have been memorized):
{interference_content}

# Instructions

Check whether the System Memories contain the interference content (or its semantic equivalent). Classify as:

- **rejected**: The interference content is NOT present in the system memories. The system correctly ignored it.
- **memorized**: The interference content IS present (or semantically equivalent) in the system memories. The system incorrectly stored it.

Use semantic matching, not exact wording.

# Output

```json
{{
  "label": "rejected|memorized",
  "reasoning": "Brief justification"
}}
```
"""

# ---------------------------------------------------------------------------
# QA Evidence Coverage (batch per question)
# ---------------------------------------------------------------------------
EVIDENCE_BATCH_PROMPT = """You are an **Evidence Retrieval Evaluator**.
Determine how many of the **Gold Evidence Points** are covered by the system's **Retrieved Memories** when answering a question.

# Inputs

1. **Retrieved Memories** (what the system retrieved to answer the question):
{retrieved_memories}

2. **Gold Evidence Points** (key facts needed for the correct answer, numbered):
{gold_evidence_points}

# Instructions

Go through each Gold Evidence Point and check whether the Retrieved Memories contain information that covers it. Semantic matching is acceptable; exact wording is NOT required.

Count how many Gold Evidence Points are **fully covered or logically implied** by the Retrieved Memories.

# Output

```json
{{
  "covered_count": <int>,
  "total": <int>,
  "reasoning": "Brief summary"
}}
```
"""

# ---------------------------------------------------------------------------
# QA Answer evaluation
# ---------------------------------------------------------------------------
QA_EVALUATION_PROMPT = """You are an **evaluation expert for AI memory system question answering**.
Based **only** on the provided **"Question"**, **"Reference Answer"**, and **"Key Memory Points"**, strictly evaluate the **accuracy** of the **"Memory System Response."** Classify it as one of **"Correct"**, **"Hallucination"**, or **"Omission."** Do **not** use any external knowledge or subjective inference.

# Evaluation Criteria

### 1. Correct
* The response accurately answers the question and is **semantically equivalent** to the Reference Answer.
* No contradictions with Key Memory Points or Reference Answer.
* Synonyms, paraphrasing, and reasonable summarization are acceptable.

### 2. Hallucination
* The response includes information that **contradicts** the Reference Answer or Key Memory Points.
* When the Reference Answer is *unknown/uncertain*, yet the response provides a specific fact.

### 3. Omission
* The response is **incomplete** compared to the Reference Answer.
* It states "don't know" or "no related memory" even though relevant information exists.
* For multi-element questions, missing **any** element counts as Omission.

## Priority Rules
* Both missing info AND fabricated info -> **Hallucination**.
* No fabrication but missing info -> **Omission**.
* Fully equivalent -> **Correct**.

# Information

* **Question:** {question}
* **Reference Answer:** {reference_answer}
* **Key Memory Points:** {key_memory_points}
* **Memory System Response:** {response}

# Output

```json
{{
  "reasoning": "Concise evaluation rationale",
  "evaluation_result": "Correct | Hallucination | Omission"
}}
```
"""

# ---------------------------------------------------------------------------
# Itemwise: per-candidate label judge (correct / hallucination / error)
# ---------------------------------------------------------------------------
MEMORY_ACCURACY_ITEMWISE_LABEL_PROMPT = """You are a Per-Memory Evaluator.

You evaluate **one** candidate memory line. Decide whether its factual claims are supported by the session dialogue and golden memory points.

# Session dialogue

{dialogue_str}

# Golden memory points (numbered)

{golden_memories_str}

# Candidate memory (the single line to evaluate)

{candidate_memory}

# Evaluation

For each factual claim in the candidate, check:
1. Is it supported by the dialogue?
2. Is it consistent with the golden memory points?

Then assign a label:

- **correct**: ALL factual claims in this candidate are supported by the dialogue and/or gold, with no conflicts or fabrications.
- **hallucination**: SOME claims are supported but SOME are not — the candidate mixes real facts with unsupported or conflicting information.
- **error**: The candidate is mostly unsupported by the dialogue, or contains obvious factual conflicts with the dialogue (e.g. wrong outcome, wrong object, reversed result).

# Output

```json
{{
  "label": "correct",
  "reasoning": "Brief justification"
}}
```

- `label`: exactly one of `"correct"`, `"hallucination"`, `"error"`.
"""

# ---------------------------------------------------------------------------
# Itemwise: per-gold recall judge (is this gold covered by any candidate?)
# ---------------------------------------------------------------------------
MEMORY_ACCURACY_ITEMWISE_RECALL_PROMPT = """You are a Gold Memory Recall Judge.

You check whether **one** golden memory point is covered by **any** of the candidate memories below.

# Golden memory point to check

{gold_memory}

# All candidate memories (numbered)

{candidate_memories_str}

# Rules

The gold is "covered" if at least one candidate conveys the same core fact — even if worded differently (paraphrase, summary, or condensation). Partial coverage also counts: if a candidate refers to the same event or action but misses some details, that still counts as covered.

The gold is "not covered" only if NO candidate has any meaningful connection to this gold fact.

# Output

```json
{{
  "covered": true,
  "matched_candidate_indices": [],
  "reasoning": "Brief justification"
}}
```

- `covered`: boolean — true if at least one candidate covers this gold point.
- `matched_candidate_indices`: 1-based candidate numbers that cover this gold (empty `[]` if none).
"""
