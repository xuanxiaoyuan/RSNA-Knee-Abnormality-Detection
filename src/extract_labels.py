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
    ("Lateral Meniscus", "tear, meaning abnormal signal that contacts the "
                         "meniscal surface. Degeneration without surface "
                         "contact = 0"),
    ("Medial OA", "moderate or large area (about 1 cm or more) of high-grade "
                  "cartilage loss (>50% thickness) in the MEDIAL tibiofemoral "
                  "compartment. Mild thinning or chondral irregularity = 0"),
    ("Lateral OA", "moderate or large area (about 1 cm or more) of high-grade "
                   "cartilage loss (>50% thickness) in the LATERAL tibiofemoral "
                   "compartment. Mild thinning or chondral irregularity = 0"),
    ("PF OA", "moderate or large area (about 1 cm or more) of high-grade "
              "cartilage loss (>50% thickness) in the PATELLOFEMORAL "
              "compartment. Mild thinning or chondral irregularity = 0"),
    ("Effusion", "moderate or large joint effusion. Small or trace effusion = 0"),
    ("Synovitis", "synovitis"),
    ("Baker's", "moderate or large Baker cyst. Small cyst = 0"),
    ("Contusion", "bone contusion, i.e. traumatic bone marrow oedema"),
    ("Fracture", "fracture"),
]

THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

# One finding per line: optional bullet/number, the finding name, a separator,
# then the verdict. Requiring the separator (":", "=", "-") and anchoring at the
# line start keeps a finding named inside another finding's *reason* from being
# mistaken for that finding's own verdict.
VERDICT_LINE = r"^[\s>*-]*(?:\d{{1,2}}\s*[.)\]:]?\s*)?{name}\s*[^\n:;]{{0,15}}?[:=\-–—]\s*([01])(?![01])"


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
        "- MOST FINDINGS ARE ABSENT. Across this dataset only about a third of "
        "the 12 slots are positive in a given report, and some findings (MCL, "
        "Baker's) are rare. Answer 1 only when the report clearly describes the "
        "finding at or above the threshold.\n"
        "- The most common error is counting a below-threshold mention as "
        "present. Words like small, mild, low-grade, trace, minimal, minor, "
        "subtle, slight, and mild degenerative change mean 0, even though the "
        "finding is mentioned. Only moderate, large, severe, high-grade, "
        "complete or full-thickness findings meet the thresholds.\n"
        "- If you are unsure whether the threshold is met, answer 0.\n"
        "- A finding that is explicitly denied (\"no fracture\") counts as 0.\n"
        "- A finding raised only as a question or a differential, and not "
        "confirmed, counts as 0.\n"
        "- Postoperative, chronic or degenerative changes count only if they "
        "meet the threshold above.\n\n"
        "Report:\n"
        '"""\n'
        f"{report}\n"
        '"""\n\n'
        "Answer with exactly 12 lines, one per finding, in the order 1-12 above, "
        "each in this format:\n"
        "1 ACL: 0 - no tear described\n"
        "2 MCL: 1 - high-grade tear\n"
        "...and so on through 12.\n\n"
        "Each line must be: the number from the list, the finding name exactly as "
        "written above, a colon, then 0 or 1, then a dash and a reason of AT MOST "
        "SIX WORDS. You are judged on the 0 or the 1, but the reason is what "
        "forces you to check whether the report actually meets the threshold - so "
        "let those few words quote what decided it.\n"
        "0 = absent, explicitly denied, or mentioned only below the threshold. "
        "1 = present at or above the threshold.\n"
        "Write all 12 lines, in order, and never skip a line. If you are running "
        "out of room, stop writing reasons and finish the remaining lines as just "
        "\"N Name: 0\" or \"N Name: 1\" - but do NOT stop before line 12.\n"
        "Do not add a summary, a conclusion, or anything after the twelfth line."
    )


def parse_answer(text, spec_names):
    """Read the per-finding verdicts out of a generation.

    Returns (bits, mode): `bits` is aligned to spec_names and holds -1 for any
    finding that could not be read, and mode is 'named' (all 12 read),
    'partial' (some) or 'fail' (none).

    Reading by NAME is the point of the format. A positional 12-character
    answer has no redundancy: one slip in the model's latent state flips several
    findings at once, which is exactly what the gold probe showed. A named line
    cannot land on the wrong finding, and unread findings stay -1 so eval
    excludes them instead of scoring them as a confident 0.
    """
    if not text:
        return None, "fail"
    cleaned = THINK_BLOCK.sub("\n", text)

    bits = {}
    for name in spec_names:
        pattern = re.compile(
            VERDICT_LINE.format(name=re.escape(name)), re.IGNORECASE | re.MULTILINE
        )
        match = pattern.search(cleaned)
        if match:
            bits[name] = int(match.group(1))

    if not bits:
        return None, "fail"
    mode = "named" if len(bits) == len(spec_names) else "partial"
    return [bits.get(name, -1) for name in spec_names], mode


def load_model(model_name, dtype_name, enable_thinking):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # A crashed run in the same kernel keeps its model alive: the retained
    # traceback holds main()'s frame, which holds the model. The next load then
    # lands on top of ~8GB that was never freed. Reporting free memory first
    # makes that failure self-evident instead of looking like a batch problem.
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB before loading")

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
    tally = {"named": 0, "partial": 0, "fail": 0}
    missing = 0
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
            bits, mode = parse_answer(text, spec_names)
            tally[mode] += 1
            if bits is None:
                bits = [-1] * len(spec_names)
            # A finding we could not read stays -1 rather than becoming a silent
            # 0.0, which would look like a confident negative.
            missing += sum(1 for bit in bits if bit < 0)
            record = {"StudyInstanceUID": study_uid}
            record.update(dict(zip(spec_names, bits)))
            record["parse_mode"] = mode
            if -1 in bits:
                record["raw_output"] = text[:300]
            rows.append(record)

        save(pd.DataFrame(rows), output_path)

    elapsed = time.time() - started
    print("[parse] " + "  ".join(f"{k}={v}" for k, v in tally.items())
          + f"  unread_cells={missing}")
    if missing:
        print("[parse] unread cells are excluded by eval_extractor.py, so a high "
              "count inflates the scorecard; inspect rows with raw_output set")
    print(f"[time] {elapsed / 60:.1f} min for {len(todo)} reports "
          f"({elapsed / max(len(todo), 1):.2f} s/report)")
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
