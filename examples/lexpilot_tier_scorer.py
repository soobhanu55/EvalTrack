"""EvalTrack scorer for an LLM: classify AI systems into EU AI Act risk tiers with a local Qwen2.5-1.5B-Instruct.

Free and local (needs `pip install torch transformers` and, for speed, a GPU). The 60 held-out descriptions are
LexPilot's evaluation set. PROMPT_VARIANT selects the prompt, which is what this example gates:

    PROMPT_VARIANT=terse   (default)  a one-line instruction
    PROMPT_VARIANT=verbose            tier definitions plus four worked examples (not taken from the 60)

The scorer reports accuracy, latency and tokens per sample, so a prompt change is judged on all three.
"""
import json
import os
import re
from pathlib import Path

from evaltrack.scorers import evaluate_llm

MODEL = os.environ.get("EVALTRACK_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
VARIANT = os.environ.get("PROMPT_VARIANT", "terse")
DATA = Path(__file__).with_name("lexpilot_eval_set.jsonl")
TIERS = ["prohibited", "high-risk", "limited-risk", "minimal-risk"]

TERSE = "Classify the AI system under the EU AI Act. Answer with exactly one of: prohibited, high-risk, limited-risk, minimal-risk."

VERBOSE = """Classify the AI system under the EU AI Act into exactly one risk tier.

Tiers:
- prohibited: banned practices (Article 5): social scoring by public authorities, manipulative or subliminal techniques that
  cause harm, exploiting the vulnerabilities of children or disabled people, real-time remote biometric identification in public
  spaces for law enforcement, untargeted scraping of facial images, emotion recognition in workplaces and schools, biometric
  categorisation by sensitive traits, predictive policing based only on profiling.
- high-risk: systems in the Annex III areas: biometric identification, critical infrastructure, education and exam scoring,
  recruitment and worker management, access to essential services such as credit scoring or benefits, law enforcement,
  migration and border control, administration of justice; and safety components of regulated products such as medical devices.
- limited-risk: systems with transparency duties: chatbots that must disclose they are AI, deepfakes and synthetic media that must
  be labelled, emotion recognition outside workplaces and schools.
- minimal-risk: everything else, such as spam filters, game AI, inventory forecasting, recommendation of films.

Examples:
Description: A bank scores loan applicants with a model that decides who gets a mortgage. -> high-risk
Description: A website chatbot answers shipping questions for a retailer. -> limited-risk
Description: A video game uses AI to control non-player characters. -> minimal-risk
Description: A city ranks citizens by behaviour and restricts services for low scorers. -> prohibited

Answer with exactly one of: prohibited, high-risk, limited-risk, minimal-risk."""

SYSTEM = {"terse": TERSE, "verbose": VERBOSE}

_model = _tok = None


def _load():
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None).eval()
    return _model, _tok


def parse(text: str) -> str:
    """The first tier named in the answer, or 'invalid'."""
    found = [(m.start(), t) for t in TIERS for m in [re.search(re.escape(t), text.lower())] if m]
    return min(found)[1] if found else "invalid"


def classify(description: str) -> tuple[str, dict]:
    import torch

    model, tok = _load()
    messages = [{"role": "system", "content": SYSTEM[VARIANT]}, {"role": "user", "content": f"Description: {description}"}]
    inputs = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    n_in = inputs["input_ids"].shape[1]
    answer = tok.decode(out[0][n_in:], skip_special_tokens=True)
    return parse(answer), {"input_tokens": n_in, "output_tokens": out.shape[1] - n_in}


def run() -> dict:
    items = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    _load()
    classify(items[0]["text"])  # warm-up call so model load and CUDA start-up do not count as latency
    result = evaluate_llm(classify, [(it["text"], it["tier"]) for it in items])
    result["prompt_variant"] = VARIANT
    return result
