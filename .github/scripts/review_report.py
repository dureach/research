#!/usr/bin/env python3
"""Independent review of a trade-intelligence report.

Proofreads the report, assigns a quality score (0-100) and an
AI-authorship score (0-100). Exits 0 on PASS, 1 on REJECT
(ai_author_score >= AI_SCORE_THRESHOLD, default 70), 2 on setup errors.

Usage: review_report.py <markdown-file>
Env:   ANTHROPIC_API_KEY (required), AI_SCORE_THRESHOLD (default 70),
       REVIEW_MODEL (default claude-sonnet-4-20250514)
Writes: review-result.json, review-summary.md (appended per file)
"""
import json
import os
import sys
import urllib.request

THRESHOLD = int(os.environ.get("AI_SCORE_THRESHOLD", "70"))
MODEL = os.environ.get("REVIEW_MODEL", "claude-sonnet-4-20250514")

PROMPT = """You are an independent senior editor at a professional trade-intelligence publication. You did not write the report below and have no stake in it. Review it cold.

Do three things:

1. PROOFREAD. List concrete, specific issues you find: grammar errors, awkward phrasing, factual inconsistencies, unclear passages, tone problems (hype, colloquialisms, marketing speak), structural problems. Quote the offending text for each issue. If an issue is minor, say so.

2. QUALITY SCORE (0-100). Judge as a professional research paper: accuracy signals, depth, clarity, structure, usefulness to a trade professional. 90+ = publishable as-is; 70-89 = solid with fixes; 50-69 = needs substantial work; below 50 = not publishable.

3. AI-AUTHORSHIP SCORE (0-100). How likely is it that this text was substantially AI-generated? Look for: formulaic transitions ("Moreover," "Furthermore," "In conclusion" stacked), repetitive sentence rhythms, generic hedging, em-dash overuse, bullet-list-heavy prose with thin narrative, vague intensifiers ("significant", "crucial", "landscape") without specifics, symmetrical/templated section structures. Score HIGH (70+) when the text reads as AI-generated; score LOW when it reads as human-written professional analysis with a distinctive voice.

VERDICT: "REJECT" if ai_author_score >= 70, otherwise "PASS".

Return ONLY valid JSON, no markdown fences, no commentary:
{"proofread_issues": ["issue 1 (quote)", "issue 2 (quote)"], "quality_score": <int>, "ai_author_score": <int>, "verdict": "PASS or REJECT", "summary": "2-3 sentence overall assessment"}

Report markdown:
---
%REPORT%
---
"""


def call_anthropic(text):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Add it in the repo's "
            "Settings -> Secrets and variables -> Actions as ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(2)
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": 4000,
            "messages": [
                {"role": "user", "content": PROMPT.replace("%REPORT%", text[:120000])}
            ],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
    except Exception as e:
        print(f"ERROR: Anthropic API call failed: {e}", file=sys.stderr)
        sys.exit(2)
    return resp["content"][0]["text"]


def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


import re  # noqa: E402  (kept after defs for single-file simplicity)


def main():
    if len(sys.argv) != 2:
        print("Usage: review_report.py <markdown-file>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path, encoding="utf-8") as f:
        text = f.read()

    raw = call_anthropic(text)
    try:
        result = parse_json(raw)
    except Exception as e:
        print(f"ERROR: could not parse reviewer JSON: {e}\nRaw output:\n{raw[:2000]}", file=sys.stderr)
        sys.exit(2)

    for k in ("proofread_issues", "quality_score", "ai_author_score", "verdict", "summary"):
        if k not in result:
            print(f"ERROR: reviewer JSON missing key '{k}'", file=sys.stderr)
            sys.exit(2)

    result["file"] = os.path.basename(os.path.dirname(os.path.abspath(path))) + "/" + os.path.basename(path)
    result["threshold"] = THRESHOLD

    with open("review-result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    verdict_icon = "REJECT" if result["verdict"] == "REJECT" else "PASS"
    lines = [
        f"## Independent review: `{result['file']}`",
        "",
        f"**Verdict: {verdict_icon}**",
        f"- Quality score: **{result['quality_score']}/100**",
        f"- AI-authorship score: **{result['ai_author_score']}/100** (reject at >= {THRESHOLD})",
        "",
        f"**Summary:** {result['summary']}",
        "",
        "**Proofread issues:**",
    ]
    issues = result["proofread_issues"] or ["None found."]
    for i, issue in enumerate(issues, 1):
        lines.append(f"{i}. {issue}")
    lines.append("")
    with open("review-summary.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    sys.exit(1 if result["verdict"] == "REJECT" else 0)


if __name__ == "__main__":
    main()
