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
    ("Medial OA", "osteoarthritis/arthrosis of the MEDIAL tibiofemoral "
                  "compartment, or a moderate/large area (~1 cm) of high-grade "
                  "cartilage loss (>50% thickness) there. An explicit "
                  "OA/chondropathy/chondromalacia label for this compartment "
                  "counts even when called mild or incipient"),
    ("Lateral OA", "osteoarthritis/arthrosis of the LATERAL tibiofemoral "
                   "compartment, or a moderate/large area (~1 cm) of high-grade "
                   "cartilage loss (>50% thickness) there. An explicit "
                   "OA/chondropathy/chondromalacia label for this compartment "
                   "counts even when called mild or incipient"),
    ("PF OA", "osteoarthritis/arthrosis of the PATELLOFEMORAL compartment, or a "
              "moderate/large area (~1 cm) of high-grade cartilage loss (>50% "
              "thickness) in the patellar/trochlear cartilage. An explicit "
              "OA/chondropathy/chondromalacia label for this compartment counts "
              "even when called mild or incipient"),
    ("Effusion", "moderate or large joint effusion. Small or trace effusion = 0"),
    ("Synovitis", "synovitis"),
    ("Baker's", "moderate or large Baker cyst. Small cyst = 0"),
    ("Contusion", "bone contusion, i.e. traumatic bone marrow oedema"),
    ("Fracture", "fracture"),
]

# The reports are multilingual and the extractor fails by RECALL: a missed
# finding is almost always one the report *does* name, in Spanish/German/Dutch/
# Turkish/..., that an English-only prompt failed to map. These are the surface
# forms that actually occur in this corpus. They are prompt vocabulary, not
# labels — the model still applies the severity threshold to what it reads.
FINDING_ALIASES = {
    "ACL": "ligamento cruzado anterior, LCA (es); vorderes Kreuzband, VKB (de); "
           "voorste kruisband (nl); ön çapraz bağ (tr); legamento crociato "
           "anteriore (it); tear = rotura/ruptur/ruptuur/yırtık/rupture/ρήξη",
    "MCL": "ligamento colateral medial, LCM (es); Innenband, mediales "
           "Kollateralband (de); mediale collaterale band (nl); medial "
           "kollateral bağ, iç yan bağ (tr)",
    "Medial Meniscus": "menisco medial, menisco interno (es); Innenmeniskus (de); "
                       "mediale meniscus (nl); medial menisküs (tr)",
    "Lateral Meniscus": "menisco lateral, menisco externo (es); Außenmeniskus "
                        "(de); laterale meniscus (nl); lateral menisküs (tr)",
    "Medial OA": "gonartrosis/artrosis femorotibial medial (es); medialer "
                 "Femorotibial-Arthrose (de); mediale artrose (nl); medial "
                 "kompartman osteoartrit (tr). cartilage loss = condropatía, "
                 "pérdida de cartílago (es); Knorpelverlust (de); "
                 "kraakbeenverlies (nl); kıkırdak kaybı (tr)",
    "Lateral OA": "artrosis femorotibial lateral (es); laterale "
                  "Femorotibial-Arthrose (de); laterale artrose (nl); lateral "
                  "kompartman osteoartrit (tr)",
    "PF OA": "artrosis femoropatelar/patelofemoral, condropatía rotuliana, "
             "condromalacia rotuliana (es); Retropatellararthrose, Chondropathie, "
             "Chondromalazie (de); patellofemorale artrose (nl); patellofemoral "
             "osteoartrit, kondromalazi patella (tr)",
    "Effusion": "derrame articular (es); Gelenkerguss (de); gewrichtsvocht, vocht "
                "(nl); eklem içi sıvı, efüzyon (tr); versamento (it); "
                "épanchement (fr); efusão, derrame (pt); αρθρικό υγρό (el)",
    "Synovitis": "sinovitis (es/it); Synovitis (de/nl); sinovit (tr); synovite "
                 "(fr); υμενίτιδα (el)",
    "Baker's": "quiste de Baker (es); Bakerzyste (de); Bakercyste (nl); Baker "
               "kisti (tr); cisti di Baker (it); kyste de Baker (fr)",
    "Contusion": "contusión ósea, edema óseo (es); Knochenkontusion, "
                 "Knochenmarködem (de); botcontusie, beenmergoedeem (nl); kemik "
                 "kontüzyonu, kemik iliği ödemi (tr)",
    "Fracture": "fractura (es/pt); Fraktur (de); fractuur (nl); fraktür, kırık "
                "(tr); frattura (it); fracture (fr); κάταγμα (el); перелом (ru)",
}

# Severity wording is the other half of the multilingual gap: "mäßig" and
# "belirgin" carry the threshold exactly as "moderate" does, and a model that
# does not recognise them quietly reads them as a below-threshold mention.
SEVERITY_BELOW = (
    "small, trace, mild, minimal, slight, subtle, low-grade (en); leve, ligero, "
    "mínimo, discreto, incipiente (es); leicht, gering, diskret, minimal (de); "
    "licht, gering, discreet, minimaal (nl); hafif, minimal, silik (tr); lieve, "
    "minimo (it); leve (pt); léger (fr)"
)
SEVERITY_AT_OR_ABOVE = (
    "moderate, large, severe, high-grade, complete, full-thickness (en); "
    "moderado, importante, marcado, severo, grave, acentuado (es); mäßig, "
    "mittelgradig, ausgeprägt, deutlich, hochgradig, schwer (de); matig, fors, "
    "uitgebreid, ernstig (nl); orta, belirgin, ileri, yoğun, ciddi (tr); "
    "moderato, marcato, grave (it); moderado, acentuado, grave (pt); modéré, "
    "important (fr)"
)

THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

# One finding per line: optional bullet/number, the finding name, a separator,
# then the verdict. Requiring the separator (":", "=", "-") and anchoring at the
# line start keeps a finding named inside another finding's *reason* from being
# mistaken for that finding's own verdict.
VERDICT_LINE = r"^[\s>*-]*(?:\d{{1,2}}\s*[.)\]:]?\s*)?{name}\s*[^\n:;]{{0,15}}?[:=\-–—]\s*([01])(?![01])"


def build_prompt(report, max_chars):
    """Build the labelling prompt.

    Two things drive the wording here. The reports arrive in a dozen languages,
    so the prompt carries the finding names and the severity words in each of
    them — the extractor's misses are overwhelmingly findings the report *does*
    name, just not in English. And "...mediaal?" (a question, not a finding)
    appears repeatedly in the Dutch reports, hence the differential rule.
    """
    listing = "\n".join(
        f"{i:>2}. {name:<18} - {desc}\n"
        f"      also written: {FINDING_ALIASES[name]}"
        for i, (name, desc) in enumerate(FINDING_SPECS, 1)
    )
    report = str(report)[:max_chars]
    return (
        "You are a musculoskeletal radiologist labelling knee MRI reports for a "
        "research dataset.\n"
        "Each report may be written in ANY language (English, Spanish, Dutch, "
        "German, Portuguese, Turkish, French, Italian, Greek, Russian, ...). Read "
        "it in its ORIGINAL language and do NOT translate it first.\n\n"
        "For each of the 12 findings, decide whether the report describes it as "
        "PRESENT under the SEVERITY THRESHOLD given. The threshold is part of the "
        "question: a finding that is mentioned but below threshold counts as 0.\n\n"
        f"{listing}\n\n"
        "Severity wording carries the verdict in every language:\n"
        f"- BELOW threshold, answer 0: {SEVERITY_BELOW}\n"
        f"- AT or ABOVE threshold, answer 1: {SEVERITY_AT_OR_ABOVE}\n\n"
        "Rules:\n"
        "- Most of the 12 slots are negative in any one report, so answer 1 only "
        "when the report clearly describes the finding at or above the threshold.\n"
        "- BUT the usual mistake - and the one to guard against - is answering 0 "
        "for a finding the report DOES name in another language. Before you write "
        "a 0, match the report's exact wording against the 'also written' list "
        "and the severity words above. If it names the finding at or above "
        "threshold, the answer is 1, whatever the language.\n"
        "- A finding is 0 when the report denies it (\"no fracture\"), never "
        "mentions it, or mentions it only below the threshold.\n"
        "- A finding raised only as a question or a differential, and not "
        "confirmed, counts as 0.\n"
        "- Postoperative, chronic or degenerative changes count only if they "
        "meet the threshold above.\n"
        "- For the three OA findings: an explicit osteoarthritis / arthrosis / OA "
        "or chondropathy / chondromalacia report for that compartment counts as "
        "1, even when the report calls it mild or incipient.\n\n"
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
        "let those few words quote what decided it, in its original language.\n"
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


def load_model(model_name, dtype_name, enable_thinking, load_in_4bit=False):
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

    # An 8-9B teacher in fp16 wants ~18GB, over a 16GB T4's headroom. nf4 cuts
    # the weights to ~5GB, which is what makes a bigger, more diverse teacher
    # affordable as one end of the ensemble. bitsandbytes is present on Kaggle;
    # if it is not, falling back to fp16 (and possibly OOMing) is better than
    # dying, but say so loudly.
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quant, device_map="auto"
            )
            model.eval()
            print(f"[model] loaded {model_name} 4-bit (nf4) on {model.device}")
            return model, tokenizer
        except Exception as exc:
            print(f"[warn] 4-bit load failed ({exc}); falling back to {dtype_name}")

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
    parser.add_argument("--output", default=None,
                        help="override the output FILENAME (joined to work_dir). "
                             "One file per teacher so several can be merged later.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override batch_size, for throughput probing")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="load the model 4-bit (nf4) so an 8-9B teacher fits a "
                             "16GB T4; needs bitsandbytes. Falls back to fp16 if absent.")
    parser.add_argument("--full", action="store_true",
                        help="force labelling ALL studies, overriding config only_gold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.model:
        cfg["model_name"] = args.model
    if args.output:
        cfg["output"] = args.output
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    load_in_4bit = bool(args.load_in_4bit) or bool(cfg.get("load_in_4bit", False))

    # The switch lives in the config, not the CLI, because the config ships in
    # the Kaggle dataset while the notebook is edited by hand on the site. A
    # probe-vs-full toggle belongs with the code that gets refreshed anyway.
    only_gold = (bool(args.only_gold) or bool(cfg.get("only_gold", False))) \
        and not args.full

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

    if only_gold:
        # Cheap quality probe: the gold studies are the only ones we can score
        # against, so labelling just those answers "is the prompt any good?"
        # in a minute instead of spending an hour before finding out.
        present = [c for c in labels if c in train.columns]
        gold_ids = set(train.loc[train[present].notna().all(axis=1), "StudyInstanceUID"])
        train = train[train["StudyInstanceUID"].isin(gold_ids)]
        print(f"[gold] restricting to the {len(gold_ids)} gold-labelled studies")

    output_path = Path(cfg.get("work_dir", "work")) / cfg.get("output", "labels_derived.csv")
    # A probe is re-run for every prompt edit, so it must NOT resume: the same
    # gold studies are already in the output from the last probe, and skipping
    # them would silently re-score the previous prompt. The full-run file is
    # overwritten, which is fine — the probe's whole output is 58 rows.
    done = read_done(output_path, bool(cfg.get("resume", True)) and not only_gold)
    seen = set(done["StudyInstanceUID"]) if len(done) else set()

    todo = train[["StudyInstanceUID", "Report"]].dropna(subset=["Report"])
    todo = todo[~todo["StudyInstanceUID"].isin(seen)]
    if cfg.get("limit"):
        todo = todo.head(int(cfg["limit"]))
    # Length-bucketed batching. A batch is padded to its LONGEST report, so
    # mixing a 300-char note with a 3900-char one makes the short ones ride
    # along at a pretence length. Sorting by length first keeps neighbours
    # similar and was the single biggest throughput cost in the last full run
    # (33 s per batch of 4). Resume is by StudyInstanceUID, so order is free.
    if len(todo):
        todo = (todo.assign(_n=todo["Report"].astype(str).str.len())
                    .sort_values("_n").drop(columns="_n"))
    print(f"[todo] {len(todo)} reports to label ({len(seen)} already done)")
    if not len(todo):
        print("[done] nothing to do")
        return

    model, tokenizer = load_model(
        cfg["model_name"], cfg.get("dtype", "float16"),
        bool(cfg.get("enable_thinking", False)), load_in_4bit,
    )

    max_chars = int(cfg.get("max_report_chars", 8000))
    batch_size = int(cfg.get("batch_size", 16))
    max_new = int(cfg.get("max_new_tokens", 32))
    print(f"[cfg] model={cfg['model_name']} batch_size={batch_size} "
          f"4bit={load_in_4bit} max_new={max_new}")

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
