# Official Test Prompts (also declared in metadata.json)

These are the exact 2 prompts submitted in `metadata.json` → `test_prompts`.
Organizers will generate 2 additional hidden prompts in the agriculture
domain to test for overfitting — all 4 are used for `S_acc` scoring.

## tp_001 — Disease diagnosis
**Prompt:** "The lower leaves of my maize plants have long grey-green
water-soaked streaks that turned tan and dry. What disease is this, and how
do I treat it?"

**Why this prompt:** Tests retrieval precision (the symptom description maps
closely to the Northern Corn Leaf Blight section in
`data/agri_docs/maize_diseases.md`) and tests that the model gives a
grounded, actionable treatment recommendation rather than a generic answer.

**Expected retrieval:** Should surface the "Northern Corn Leaf Blight"
section as the top-scoring BM25 result.

## tp_002 — Fertilizer guidance
**Prompt:** "I am planting maize this season on a small plot with average
soil fertility. What fertilizer type and quantity should I apply at
planting, and when should I top-dress?"

**Why this prompt:** Tests that the model correctly separates basal
(planting-time) application from top-dressing timing, and gives concrete
quantities rather than vague advice — directly exercising the "Basal" and
"Top-Dressing" sections of `data/agri_docs/fertilizer_recommendations.md`.

**Expected retrieval:** Should surface the "Basal (Planting-Time)
Application" and "Top-Dressing (Nitrogen)" sections as top results.

## Manual verification
Run either prompt directly with:

```bash
python -m src.app --prompt "The lower leaves of my maize plants have long grey-green water-soaked streaks that turned tan and dry. What disease is this, and how do I treat it?"
```

Confirm the printed "Sources" line references the expected section before
submitting.
