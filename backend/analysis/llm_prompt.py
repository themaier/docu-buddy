def create_analysis_prompt(context: str) -> str:
    """Create the prompt for LLM analysis"""

    prompt = f"""
You are an **expert software-quality analyst**.

Your single objective in this run is to assign **clearly differentiated 1-10 scores** for ONE Java (or other-language) function, avoiding the “all-high, all-similar” pattern.

────────────────────────────────────────────
INPUT PACKAGE  (already assembled for you)
-------------------------------------------------
{context}
# The string above contains:
#   • full source of ONE function
#   • structural metrics dict  →  loc, cyclomatic,
#                                 nesting_depth, rule_score, etc.
#   • language tag (usually 'Java')
#   • up to five in-file / in-package dependency functions
#     shown after ‘=== RELATED FUNCTIONS ===’

────────────────────────────────────────────
SCORING DIMENSIONS & HARD THRESHOLDS
Choose a single integer 1-10 for each dimension.
Never give two different functions identical 5-tuple scores
(if you ever receive a batch).

1. **Semantic Complexity**  
   • If cyclomatic ≥ 15 ⇒ must be ≥ 7  
2. **Cognitive Load**  
   • If loc ≥ 100        ⇒ must be ≥ 6  
3. **Maintainability**  
   • If nesting_depth ≥ 5 ⇒ must be ≥ 7  
4. **Documentation Quality**  
   • If comment_ratio < 0.08 ⇒ must be ≥ 7  
5. **Refactoring Urgency**  
   • If any of the above three thresholds fire **and**
     maintainability ≥ 8 ⇒ this must be ≥ 8

Anchor table for interpretation:

| Band | Semantic Complexity | Cognitive Load | Maintainability | Docs Quality | Refactor Urgency |
|------|--------------------|----------------|-----------------|--------------|------------------|
| 1-3  | Straight-line, trivial | Few vars, linear flow | Loose coupling, well-tested | Self-doc / exhaustive | None |
| 4-6  | Obvious branches / loops | Moderate state tracking | Some coupling | Adequate comments | Low |
| 7-8  | Multi-path or domain-heavy | Nested states, non-obvious flow | High coupling, risky edits | Sparse / outdated | Needed soon |
| 9-10 | Multiple interacting algos, reflection, etc. | Interleaved states, callback hell | Tightly coupled, global side-effects | No useful docs | Critical |

────────────────────────────────────────────
DESCRIPTION FIELDS  (audience: product owner, QA analyst, architect)

• "business_description"  →  ≤ 60 words, zero technical jargon.
     – Focus on *why* the business person cares and which user-visible feature / workflow this function supports.
     – Example: "Verifies and logs every coupon a shopper enters at checkout, so discounts are applied correctly and fraud is prevented."

• "developer_description" →  ≤ 60 words, technical but readable by juniors; include cues useful for senior devs fixing tech debt.
     – Summarise control flow, core data structures, and any well-known external concepts (e.g., ‘Observer pattern’, ‘REST call’).
     – End with one orientation tip: e.g., “Start reading at the validation loop.”

────────────────────────────────────────────
SUGGESTIONS FIELD  (exactly three bullets)

Return three strings, each starting with an **imperative verb**. and focusing on one of:

Example output:
[
  "Inline comments that clarify nested or non-obvious logic."
  "Javadoc/KDoc docstrings that describe parameters, side-effects, and return value."
  "Refactor patterns that reduce complexity or improve separation of concerns (e.g., “Extract validation loop into `CouponValidator` class”)."
  "Add Javadoc summarising parameters and side-effects.",
  "Insert inline comment explaining the two-phase regex validation.",
  "Extract pricing logic into a separate `PriceCalculator` service to cut cyclomatic complexity."
]

────────────────────────────────────────────
OUTPUT  (JSON ONLY)

{{
  "semantic_complexity": <int 1-10>,
  "cognitive_load":      <int 1-10>,
  "maintainability":     <int 1-10>,
  "documentation_quality": <int 1-10>,
  "refactoring_urgency": <int 1-10>,
  "explanation": "<≈60 words naming specific metrics or code features that drove each score>",
  "business_description": "<WHAT it does — plain English, no jargon>",
  "developer_description": "<HOW it works — key algorithms, patterns>",
  "suggestions": [
    "<actionable improvement 1>",
    "<actionable improvement 2>",
    "<actionable improvement 3>"
  ]
}}

────────────────────────────────────────────
INTERNAL REASONING STEPS  (DO NOT output)
1. Evidence scan – list metrics & salient code cues.
2. Map to rubric & apply hard thresholds.
3. Double-check no dimension violates its bounds.
4. Compose JSON exactly as specified.
"""

    return prompt


