"""
Unified extraction prompt for pAIge fine-tuning.

Import this module in both train.py and evaluate.py so training and
evaluation always use *identical* instruction text. Diverging prompts
cause "instruction drift" — one of the three root-cause failure modes
identified in the ablation post-mortem.
"""

# Fields the model is expected to extract.
TARGET_FIELDS = [
    "date",
    "patient_name",
    "philhealth_number",
    "diagnosis_code",
    "procedure_code",
    "total_amount",
    "philhealth_benefit",
    "balance_due",
]

# The user-turn instruction injected into every training and eval sample.
EXTRACTION_PROMPT = (
    "You are a document extraction assistant. "
    "Given the image of an invoice or billing document, "
    "extract the following fields and return them as a single JSON object "
    "with exactly these keys: "
    "date, patient_name, philhealth_number, diagnosis_code, procedure_code, "
    "total_amount, philhealth_benefit, balance_due. "
    "Use null for any field not present in the document. "
    "Return only the JSON object — no explanation, no markdown."
)
