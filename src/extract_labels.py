"""Turn the multilingual reports into 12 binary labels per study.

Only 58 of 4407 training studies carry labels, so the image model has nothing
to learn from unless we derive labels for the rest from their reports. This
does that with an open-weights multilingual LLM.

The severity thresholds in the prompt are not decoration. The gold labels were
set by radiologists reading images against explicit severity criteria, and the
reports routinely mention findings that fall below them — "small joint
effusion", "mild cartilage thinning". A prompt that only asks "was it
mentioned?" gets those systematically wrong.

Run in a Kaggle notebook with Internet ON (weights download from HuggingFace).
The output CSV is a one-time artefact; the offline submission only reads it.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import discover_labels, load_config, read_metadata, resolve_data_dir  # noqa: E402

# The order here defines the order of the 12 bits the model must emit.
# It is also the column order of the output file.
FINDING_SPECS = [
    ("ACL", "high-grade partial or full-thickness tear (>50% of fibres). "
            "Mild signal change or degeneration only = 0"),
    ("MCL", "high-grade partial or complete acute tear. "
            "Low-grade sprain or chronic change = 0"),
    ("Medial Meniscus", "tear, meaning abnormal signal that contacts the "
                        "meniscal surface. Degeneration without surface "
                        "contact = 0"),
    ("Lateral Meniscus", "same criterion as the medial meniscus"),
    ("Medial OA", "moderate or large area (about 1 cm or more) of high-grade "
                  "cartilage loss (>50% thickness) in the MEDIAL tibiofemoral "
                  "compartment. Mild thinning or chondral irregularity = 0"),
    ("Lateral OA", "same criterion as Medial OA, in the LATERAL tibiofemoral "
                   "compartment"),
    ("PF OA", "same criterion as Medial OA, in the PATELLOFEMORAL compartment"),
    ("Effusion", "moderate or large joint effusion. Small or trace effusion = 0"),
    ("Synovitis", "synovitis"),
    ("Baker's", "moderate or large Baker cyst. Small cyst = 0"),
    ("Contusion", "bone contusion, i.e. traumatic bone marrow oedema"),
    ("Fracture", "fracture"),
]

THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
EXACT_BITS = re.compile(r"(?<![01])[01]{12}(?![01])")
LOOSE_BITS = re.compile(r"[01]{12}")
NUMERIC_ANSWER = re.compile(r"[01]{1,12}(?:\.0+)?")


def build_prompt(report, max_chars):
    """Build the labelling prompt.

    The language instruction and the negation rule both matter: the reports
    arrive in a dozen languages, and "...mediaal?" (a question, not a finding)
    appears repeatedly in the Dutch ones.
    """
    listing = "\n".join(
        f"{i:>2}. {name:<18} - {desc}" for i, (name, desc) in enumerate(FINDING_SPECS, 1)
    )
    report = str(report)[:max_chars]
    return (
        "You are a musculoskeletal radiologist labelling knee MRI reports for a "
        "research dataset.\n"
        "Each report may be written in ANY language (English, Spanish, Dutch, "
        "German, Portuguese, Turkish, French, ...). Read it in its original "
        "language. Do NOT translate it first.\n\n"
        "For each of the 12 findings, decide whether the report describes it as "
        "PRESENT under the SEVERITY THRESHOLD given. The threshold is part of the "
        "question: a finding that is mentioned but below threshold counts as 0.\n\n"
        f"{listing}\n\n"
        "Rules:\n"
        "- A finding that is explicitly denied (\"no fracture\") counts as 0.\n"
        "- A finding raised only as a question or a differential, and not "
        "confirmed, counts as 0.\n"
        "- Postoperative, chronic or degenerative changes count only if they "
        "meet the threshold above.\n\n"
        "Report:\n"
        '"""\n'
        f"{report}\n"
        '"""\n\n'
        "Answer with exactly 12 characters, each 0 or 1, in the order 1-12 above.\n"
        "Write out all 12 characters, including any leading zeros. This is a "
        "STRING of characters, not a number: never drop leading zeros, never "
        "write a decimal point or a trailing .0.\n"
        "For example, if findings 3 and 11 are present and the other ten are "
        "absent, the answer is exactly 001000000010\n"
        "Output nothing else: no explanation, no spaces, no punctuation."
    )


def parse_answer(text):
    """Pull the 12 bits out of a generation.

    Returns (bits, mode) where mode is 'exact', 'padded', 'loose' or 'fail'.
    Anything but 'exact' is an inference rather than a clean read, so the modes
    are counted separately and can be inspected instead of silently trusted.
    """
    if not text:
        return None, "fail"
    cleaned = THINK_BLOCK.sub(" ", text)
    match = EXACT_BITS.search(cleaned)
    if match:
        return [int(c) for c in match.group(0)], "exact"

    # The model sometimes renders the 12 bits as a NUMBER, which drops leading
    # zeros and appends ".0" -- 001011110100 comes back as "1011110100.0".
    # Recover that only when the entire answer is the number: a looser rule
    # would happily mine 0s and 1s out of prose.
    bare = cleaned.strip()
    if NUMERIC_ANSWER.fullmatch(bare):
        digits = bare.split(".")[0]
        return [int(c) for c in digits.rjust(12, "0")], "padded"

    digits = re.sub(r"[^01]", "", cleaned)
    fallback = LOOSE_BITS.search(digits)
    if fallback:
        return [int(c) for c in fallback.group(0)], "loose"
    return None, "fail"


def load_model(model_name, dtype_name, enable_thinking):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        dtype_name, torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Left padding keeps the generated continuation at a consistent position
    # across a padded batch, which batched decoding requires.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    print(f"[model] loaded {model_name} as {dtype_name} on {model.device}")
    return model, tokenizer


def render_prompts(tokenizer, prompts, enable_thinking):
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        except TypeError:
            # Older chat templates have no thinking toggle; the parser copes.
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
    return texts


def read_done(output_path, resume):
    """Studies already labelled, so a killed session resumes instead of restarting."""
    if not resume or not Path(output_path).exists():
        return pd.DataFrame()
    done = pd.read_csv(output_path)
    if "StudyInstanceUID" not in done.columns:
        return pd.DataFrame()
    print(f"[resume] {len(done)} studies already labelled in {output_path}")
    return done


def save(frame, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extract.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="only label the first N studies (smoke test)")
    parser.add_argument("--only-gold", action="store_true",
                        help="label only the studies that already carry gold labels, "
                             "for a one-minute quality probe via eval_extractor.py")
    parser.add_argument("--model", default=None, help="override model_name")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.model:
        cfg["model_name"] = args.model

    import torch
    from tqdm import tqdm

    data_dir = resolve_data_dir(cfg, search_roots=("/kaggle/input", "."))
    labels = discover_labels(data_dir)
    train = read_metadata(data_dir, "train.csv")
    if train is None or "Report" not in train.columns:
        raise FileNotFoundError("train.csv with a Report column is required")

    spec_names = [name for name, _ in FINDING_SPECS]
    if spec_names != labels:
        print(f"[warn] finding order differs from submission columns")
        print(f"[warn]   prompt order: {spec_names}")
        print(f"[warn]   submission:   {labels}")

    if args.only_gold:
        # Cheap quality probe: the gold studies are the only ones we can score
        # against, so labelling just those answers "is the prompt any good?"
        # in a minute instead of spending an hour before finding out.
        present = [c for c in labels if c in train.columns]
        gold_ids = set(train.loc[train[present].notna().all(axis=1), "StudyInstanceUID"])
        train = train[train["StudyInstanceUID"].isin(gold_ids)]
        print(f"[gold] restricting to the {len(gold_ids)} gold-labelled studies")

    output_path = Path(cfg.get("work_dir", "work")) / cfg.get("output", "labels_derived.csv")
    done = read_done(output_path, bool(cfg.get("resume", True)))
    seen = set(done["StudyInstanceUID"]) if len(done) else set()

    todo = train[["StudyInstanceUID", "Report"]].dropna(subset=["Report"])
    todo = todo[~todo["StudyInstanceUID"].isin(seen)]
    if cfg.get("limit"):
        todo = todo.head(int(cfg["limit"]))
    print(f"[todo] {len(todo)} reports to label ({len(seen)} already done)")
    if not len(todo):
        print("[done] nothing to do")
        return

    model, tokenizer = load_model(
        cfg["model_name"], cfg.get("dtype", "float16"), bool(cfg.get("enable_thinking", False))
    )

    max_chars = int(cfg.get("max_report_chars", 8000))
    batch_size = int(cfg.get("batch_size", 16))
    max_new = int(cfg.get("max_new_tokens", 32))

    rows = list(done.to_dict("records")) if len(done) else []
    tally = {"exact": 0, "padded": 0, "loose": 0, "fail": 0}
    started = time.time()

    for start in tqdm(range(0, len(todo), batch_size), desc="labelling"):
        chunk = todo.iloc[start:start + batch_size]
        prompts = [build_prompt(r, max_chars) for r in chunk["Report"]]
        texts = render_prompts(tokenizer, prompts, bool(cfg.get("enable_thinking", False)))
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for study_uid, text in zip(chunk["StudyInstanceUID"], decoded):
            bits, mode = parse_answer(text)
            tally[mode] += 1
            record = {"StudyInstanceUID": study_uid}
            if bits is None:
                # Leave the row in with an explicit marker rather than a silent
                # 0.0, which would look like a confident negative.
                record.update({name: -1 for name in spec_names})
                record["parse_mode"] = "fail"
                record["raw_output"] = text[:200]
            else:
                record.update(dict(zip(spec_names, bits)))
                record["parse_mode"] = mode
            rows.append(record)

        save(pd.DataFrame(rows), output_path)

    elapsed = time.time() - started
    print("[parse] " + "  ".join(f"{k}={v}" for k, v in tally.items()))
    if tally["fail"] or tally["loose"] or tally["padded"]:
        print("[parse] inspect labels_derived.csv rows where parse_mode != 'exact'")
    print(f"[time] {elapsed / 60:.1f} min for {len(todo)} reports "
          f"({elapsed / max(len(todo), 1):.2f} s/report)")
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
