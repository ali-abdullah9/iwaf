"""
parse_modsec_audit.py — Parse OWASP ModSecurity audit logs to a single CSV.

Walks DATA_DIR/<DD-Mon-YYYY>/modsec_audit.anon.log (one file per day),
extracts each ModSec transaction, and writes a unified CSV with columns:

  method, path, query_string, body, label,
  source_ip, timestamp, modsec_verdict, rule_id, msg

Schema notes
------------
- The first 5 cols match unified_v2/train.csv so existing code (data_utils,
  IWAFDataset, etc.) keeps working.
- The last 5 cols are *new* and are what the GCN re-evaluation needs.
- `label` is DERIVED from the OWASP CRS rule ID prefix (see CRS_RANGE_LABEL).
  Transactions ModSec did not block are labelled `benign` and given a
  `modsec_verdict='allowed'` so you can filter as needed.
- Custom site rule IDs (e.g. 444444 BadBot) outside the CRS ranges are
  bucketed as `probe`.

ModSec audit log format reference
---------------------------------
Each transaction is delimited by section markers --<unique_id>-A-- ... --Z--
Sections we use:
  A : timestamp + client IP + ports + server IP
  B : full HTTP request (method, path, headers, ?body for non-multipart)
  C : request body when present (multipart, large bodies)
  H : ModSecurity verdict + matched rules

CLI
---
  python parse_modsec_audit.py --data-dir owasp_raw --out zenodo_with_meta.csv
  python parse_modsec_audit.py --data-dir owasp_raw --out small.csv --max-days 3
"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime
from urllib.parse import urlsplit


# --------------------------------------------------------------------------
# CRS rule-id prefix → attack class
# Derived from coreruleset v3 rule files (rules/REQUEST-9XX-*.conf).
# Documented here so the mapping is auditable:
# --------------------------------------------------------------------------
CRS_RANGE_LABEL = {
    "913": "probe",   # scanner detection
    "920": "probe",   # protocol enforcement
    "921": "probe",   # protocol attack
    "930": "lfi",     # path traversal / LFI
    "931": "lfi",     # RFI
    "932": "cmdi",    # RCE / unix shell injection
    "933": "cmdi",    # PHP / code injection
    "941": "xss",
    "942": "sqli",
    "943": "sqli",    # session fixation (treated as SQLi family)
    "944": "cmdi",    # java RCE
}

VALID_LABELS = ("benign", "sqli", "xss", "lfi", "cmdi", "probe")


# --------------------------------------------------------------------------
# Section split: looks for `--<id>-X--` markers and groups by transaction id
# --------------------------------------------------------------------------
SECTION_RE = re.compile(r"^--([0-9a-f]{6,12})-([A-Z])--\s*$", re.MULTILINE)


def parse_a_section(text):
    """A-section: `[27/Jul/2025:00:00:22 +0200] <unique_id> <client_ip> <client_port> <server_ip> <server_port>`"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    line = lines[0] if lines else ""
    m = re.match(
        r"^\[([^\]]+)\]\s+\S+\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)$", line
    )
    if not m:
        return None
    ts_raw, src_ip, _src_port, _dst_ip, _dst_port = m.groups()
    try:
        ts = datetime.strptime(ts_raw, "%d/%b/%Y:%H:%M:%S %z")
        ts_iso = ts.isoformat()
    except ValueError:
        ts_iso = ts_raw
    return {"timestamp": ts_iso, "source_ip": src_ip}


def parse_b_section(text):
    """B-section: full HTTP request. Lines: `METHOD path?query HTTP/1.x` then headers."""
    # Section bodies always have a leading newline from the section delimiter,
    # so the first non-empty line is the request line.
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    method = path = query = ""
    body = ""
    if not lines:
        return {"method": method, "path": path, "query_string": query, "body": body}
    first = lines[0]
    m = re.match(r"^(\S+)\s+(\S+)\s+HTTP/\d+\.\d+\s*$", first)
    if m:
        method = m.group(1)
        url = m.group(2)
        # Split path from query string. urlsplit handles things like
        # `/foo?a=1&b=2`.  ModSec does NOT include scheme/host here.
        sp = urlsplit(url)
        path = sp.path or url
        query = sp.query or ""

    # Body in B-section appears AFTER a blank line if present (rare; usually
    # body lives in C-section).  Walk until blank, then collect the rest.
    if len(lines) > 1 and "" in lines[1:]:
        idx = lines[1:].index("")  # first blank line
        body_lines = lines[1 + idx + 1:]
        body = "\n".join(body_lines).strip()

    return {"method": method, "path": path, "query_string": query, "body": body}


def parse_h_section(text):
    """H-section: extract verdict + the most informative rule ID.

    Strategy: collect ALL [id "..."] tags. Prefer a CRS rule (913-944
    prefixes) over any custom site rule (like 444444 BadBot), because CRS
    rules carry the attack-class signal we need for labelling.
    """
    blocked = "Access denied" in text or "Intercepted" in text
    all_ids = re.findall(r'\[id "(\d+)"\]', text)
    msg_matches = re.findall(r'\[msg "([^"]+)"\]', text)

    # Prefer CRS rule (any id whose prefix is in CRS_RANGE_LABEL); else fallback
    # to whatever rule appeared (often a custom site rule).
    rule_id = ""
    msg = ""
    crs_ids = [r for r in all_ids if r[:3] in CRS_RANGE_LABEL]
    if crs_ids:
        rule_id = crs_ids[0]
        # find msg paired with this id (best-effort: same index in original order)
        try:
            idx = all_ids.index(rule_id)
            msg = msg_matches[idx] if idx < len(msg_matches) else (msg_matches[0] if msg_matches else "")
        except ValueError:
            msg = msg_matches[0] if msg_matches else ""
    elif all_ids:
        rule_id = all_ids[0]
        msg = msg_matches[0] if msg_matches else ""

    return {
        "modsec_verdict": "blocked" if blocked else "allowed",
        "rule_id": rule_id,
        "msg": msg,
        "all_rule_ids": ",".join(sorted(set(all_ids))),
    }


def label_from_rule(rule_id):
    """Derive attack-class label from a CRS rule ID. Empty/non-CRS → probe."""
    if not rule_id:
        return ""
    prefix = rule_id[:3]
    return CRS_RANGE_LABEL.get(prefix, "probe")


# --------------------------------------------------------------------------
# Walk one log file and yield transaction dicts
# --------------------------------------------------------------------------
def parse_log_file(path):
    """Yield one dict per transaction. Tolerant of malformed sections."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            blob = f.read()
    except OSError as e:
        print(f"  skip {path}: {e}", file=sys.stderr)
        return

    # Split by section markers. The result is interleaved:
    # [pre-text, txn_id_1, sec_letter_1, body_1, txn_id_2, sec_letter_2, ...]
    parts = SECTION_RE.split(blob)

    # Group sections by transaction id
    txns = {}        # txn_id -> {section_letter: body_text}
    order = []       # preserve first-seen order of transactions
    i = 1
    while i + 2 < len(parts):
        txn_id = parts[i]
        letter = parts[i + 1]
        body = parts[i + 2]
        if txn_id not in txns:
            txns[txn_id] = {}
            order.append(txn_id)
        txns[txn_id][letter] = body
        i += 3

    for txn_id in order:
        sections = txns[txn_id]
        a = parse_a_section(sections.get("A", ""))
        if a is None:
            continue                      # malformed / non-A-leading transaction
        b = parse_b_section(sections.get("B", ""))
        if not b["method"]:
            continue                      # no parseable request line
        # body sometimes lives in C
        if not b["body"] and "C" in sections:
            b["body"] = sections["C"].strip()
        h = parse_h_section(sections.get("H", ""))

        # Labelling logic:
        #   1. If a CRS rule (913-944 prefixes) fired, label by that rule's
        #      class — regardless of whether ModSec ultimately blocked.
        #      (CRS in anomaly-scoring mode often matches without blocking.)
        #   2. If only a custom site rule fired (e.g. 444444 BadBot), label probe.
        #   3. If no rule fired at all, label benign.
        crs_rule = h["rule_id"] if h["rule_id"] and h["rule_id"][:3] in CRS_RANGE_LABEL else ""
        if crs_rule:
            label = CRS_RANGE_LABEL[crs_rule[:3]]
        elif h["rule_id"]:
            label = "probe"          # custom site rule, e.g. BadBot
        else:
            label = "benign"

        yield {
            "method": b["method"],
            "path": b["path"],
            "query_string": b["query_string"],
            "body": b["body"][:8000],     # cap body to avoid 10MB rows
            "label": label,
            "source_ip": a["source_ip"],
            "timestamp": a["timestamp"],
            "modsec_verdict": h["modsec_verdict"],
            "rule_id": h["rule_id"],
            "msg": h["msg"],
            "all_rule_ids": h["all_rule_ids"],
        }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Directory containing <DD-Mon-YYYY>/modsec_audit.anon.log")
    ap.add_argument("--out", default="zenodo_with_meta.csv")
    ap.add_argument("--max-days", type=int, default=None,
                    help="Only parse this many day-folders (for quick tests)")
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.data_dir, "*", "modsec_audit.anon.log")))
    if args.max_days:
        logs = logs[: args.max_days]
    if not logs:
        sys.exit(f"No logs found under {args.data_dir}/*/modsec_audit.anon.log")

    print(f"Parsing {len(logs)} log file(s)...")
    fields = [
        "method", "path", "query_string", "body", "label",
        "source_ip", "timestamp", "modsec_verdict", "rule_id", "msg",
        "all_rule_ids",
    ]

    n_total = 0
    n_blocked = 0
    label_counts = {}
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for log_path in logs:
            day = os.path.basename(os.path.dirname(log_path))
            n_day = 0
            for row in parse_log_file(log_path):
                w.writerow(row)
                n_total += 1
                n_day += 1
                if row["modsec_verdict"] == "blocked":
                    n_blocked += 1
                label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
            print(f"  {day}: {n_day:,} transactions")

    print()
    print(f"Total transactions written : {n_total:,}")
    print(f"  ModSec verdict=blocked   : {n_blocked:,}")
    print(f"  ModSec verdict=allowed   : {n_total - n_blocked:,}")
    print(f"\nLabel distribution (by CRS rule class, not by block-verdict):")
    for lbl in VALID_LABELS:
        print(f"  {lbl:8s}: {label_counts.get(lbl, 0):,}")
    print(f"\nWrote → {args.out}")


if __name__ == "__main__":
    main()
