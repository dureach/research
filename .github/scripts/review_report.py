#!/usr/bin/env python3
"""Independent review of a trade-intelligence report.

Proofreads the report, assigns a quality score (0-100) and an
AI-authorship score (0-100). Exits 0 on PASS, 1 on REJECT
(ai_author_score > AI_REJECT_ABOVE or quality_score < QUALITY_MIN_SCORE),
2 on setup errors.

Usage: review_report.py <markdown-file>
Env:   REVIEW_PROVIDER (deepseek | openrouter, default deepseek)
       DEEPSEEK_API_KEY (required when provider=deepseek)
       OPENROUTER_API_KEY (required when provider=openrouter)
       AI_REJECT_ABOVE (default 20), QUALITY_MIN_SCORE (default 90),
       REVIEW_MODEL (default deepseek-flash | nvidia/nemotron-3-ultra-550b-a55b:free)
Writes: review-result.json, review-summary.md (appended per file)
"""
import json
import os
import re
import sys
import urllib.request

AI_REJECT_ABOVE = int(os.environ.get("AI_REJECT_ABOVE", "20"))
QUALITY_MIN = int(os.environ.get("QUALITY_MIN_SCORE", "90"))
PROVIDER = os.environ.get("REVIEW_PROVIDER", "deepseek").lower()

PROVIDERS = {
    "deepseek": {
        "url": os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"),
        "key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-flash",
        "json_mode": True,
    },
    "openrouter": {
        "url": os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "json_mode": False,
    },
}

if PROVIDER not in PROVIDERS:
    print(f"ERROR: unknown REVIEW_PROVIDER '{PROVIDER}' (use deepseek or openrouter)", file=sys.stderr)
    sys.exit(2)

CFG = PROVIDERS[PROVIDER]
MODEL = os.environ.get("REVIEW_MODEL", CFG["default_model"])
API_URL = CFG["url"]

PROMPT = """You are an independent senior editor at a professional trade-intelligence publication. You did not write the report below and have no stake in it. Review it cold.

Do three things:

1. PROOFREAD. List concrete, specific issues you find: grammar errors, awkward phrasing, factual inconsistencies, unclear passages, tone problems (hype, colloquialisms, marketing speak), structural problems. Quote the offending text for each issue. If an issue is minor, say so.

2. QUALITY SCORE (0-100). Judge as a professional research paper: accuracy signals, depth, clarity, structure, usefulness to a trade professional. 90+ = publishable as-is; 70-89 = solid with fixes; 50-69 = needs substantial work; below 50 = not publishable.

3. AI-AUTHORSHIP SCORE (0-100). How likely is it that this text was substantially AI-generated? Look for: formulaic transitions ("Moreover," "Furthermore," "In conclusion" stacked), repetitive sentence rhythms, generic hedging, em-dash overuse, bullet-list-heavy prose with thin narrative, vague intensifiers ("significant", "crucial", "landscape") without specifics, symmetrical/templated section structures. Score HIGH (70+) when the text reads as AI-generated; score LOW when it reads as human-written professional analysis with a distinctive voice.

VERDICT: "REJECT" if ai_author_score > 20 or quality_score < 90, otherwise "PASS".

CRITICAL: do NOT narrate your thinking, outline your plan, or explain your process. Your ENTIRE response must be exactly one JSON object and nothing else — no preamble, no reasoning text, no trailing commentary.

Return ONLY valid JSON, no markdown fences, no commentary:
{"proofread_issues": ["issue 1 (quote)", "issue 2 (quote)"], "quality_score": <int>, "ai_author_score": <int>, "verdict": "PASS or REJECT", "summary": "2-3 sentence overall assessment"}

Report markdown:
---
%REPORT%
---
"""


def api_post(payload, headers):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(API_URL, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def call_llm(text):
    key = os.environ.get(CFG["key_env"])
    if not key:
        print(
            f"ERROR: {CFG['key_env']} is not set. Add it in the repo's "
            f"Settings -> Secrets and variables -> Actions as {CFG['key_env']}.",
            file=sys.stderr,
        )
        sys.exit(2)
    base = {
        "model": MODEL,
        "max_tokens": 4000,
        "messages": [
            {"role": "user", "content": PROMPT.replace("%REPORT%", text[:120000])}
        ],
    }
    headers = {
        "Authorization": "Bearer " + key,
        "content-type": "application/json",
    }
    if PROVIDER == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/dureach/research"
        headers["X-Title"] = "Dureach research report review"
    # Payload variants to try in order. DeepSeek mandates json_object.
    # OpenRouter: attempt structured output first (much more reliable JSON
    # compliance), fall back to a plain request if the model rejects
    # response_format with HTTP 400.
    variants = []
    if CFG["json_mode"] or PROVIDER == "openrouter":
        variants.append(dict(base, response_format={"type": "json_object"}))
    if PROVIDER == "openrouter":
        variants.append(dict(base))
    last_err = None
    for payload in variants:
        try:
            resp = api_post(payload, headers)
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 400 and payload is not variants[-1]:
                last_err = f"HTTP 400 for response_format, retrying plain: {e}"
                print(f"WARNING: {last_err}", file=sys.stderr)
                continue
            print(f"ERROR: {PROVIDER} API call failed: HTTP {e.code}: {e}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            print(f"ERROR: {PROVIDER} API call failed: {e}", file=sys.stderr)
            sys.exit(2)
    print(f"ERROR: {PROVIDER} API call failed: {last_err}", file=sys.stderr)
    sys.exit(2)


def parse_json(raw):
    """Pull the JSON object out of a model response, tolerating fences and prose.

    Some models (notably reasoning-flavored ones) emit chain-of-thought text
    before/around the JSON. We try, in order: direct parse, fenced block,
    last {...} span, then a balanced-brace scan from the first '{'.
    """
    text = raw.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            pass
                        break
    raise ValueError("no JSON object found in model output")


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

    raw = None
    result = None
    last_err = ""
    for attempt in range(1, 4):
        raw = call_llm(text)
        try:
            result = parse_json(raw)
            break
        except Exception as e:
            last_err = str(e)
            print(f"WARNING: reviewer output unparseable (attempt {attempt}/3): {last_err}", file=sys.stderr)
    if result is None:
        print(f"ERROR: could not parse reviewer JSON after 3 attempts: {last_err}\nRaw output:\n{raw[:2000]}", file=sys.stderr)
        sys.exit(2)

    for k in ("proofread_issues", "quality_score", "ai_author_score", "verdict", "summary"):
        if k not in result:
            print(f"ERROR: reviewer JSON missing key '{k}'", file=sys.stderr)
            sys.exit(2)

    result["file"] = os.path.basename(os.path.dirname(os.path.abspath(path))) + "/" + os.path.basename(path)
    result["ai_reject_above"] = AI_REJECT_ABOVE
    result["quality_min_score"] = QUALITY_MIN
    # The script — not the model — owns the verdict, so the gate can't drift.
    result["verdict"] = (
        "REJECT"
        if (result["ai_author_score"] > AI_REJECT_ABOVE or result["quality_score"] < QUALITY_MIN)
        else "PASS"
    )

    with open("review-result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    verdict_icon = "REJECT" if result["verdict"] == "REJECT" else "PASS"
    lines = [
        f"## Independent review: `{result['file']}`",
        "",
        f"**Verdict: {verdict_icon}**",
        f"- Quality score: **{result['quality_score']}/100** (reject below {QUALITY_MIN})",
        f"- AI-authorship score: **{result['ai_author_score']}/100** (reject above {AI_REJECT_ABOVE})",
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
