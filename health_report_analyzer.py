"""
health_report_analyzer.py
─────────────────────────
Core logic module for the Health Report Analyzer.
Contains: env loading, Mistral client, all agents, pipeline orchestrator.
No Streamlit or FastAPI imports — this is a pure library.
"""

import os
import json
import re
import io
import time
import pdfplumber
import pytesseract
from PIL import Image, ImageFilter
from openai import OpenAI
from dotenv import load_dotenv
from pytesseract import TesseractNotFoundError

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
TESSERACT_PATH = os.getenv("TESSERACT_PATH")

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise RuntimeError(
        "MISTRAL_API_KEY not found. "
        "Create a .env file in the project root with: MISTRAL_API_KEY=your_key_here"
    )

client = OpenAI(
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1",
)


# ═════════════════════════════════════════════════════════════════════════════
#  REFERENCE RANGES
# ═════════════════════════════════════════════════════════════════════════════
REFERENCE_RANGES = {
    "hemoglobin":       {"male": (13.5, 17.5), "female": (12.0, 15.5), "unit": "g/dL"},
    "hematocrit":       {"male": (38.3, 48.6), "female": (35.5, 44.9), "unit": "%"},
    "rbc":              {"male": (4.7, 6.1),   "female": (4.2, 5.4),   "unit": "million/µL"},
    "wbc":              {"normal": (4000, 11000), "unit": "cells/µL"},
    "platelets":        {"normal": (150000, 400000), "unit": "cells/µL"},
    "cholesterol":      {"normal": (0, 200),   "unit": "mg/dL"},
    "ldl":              {"normal": (0, 100),    "unit": "mg/dL"},
    "hdl":              {"male": (40, 999),     "female": (50, 999), "unit": "mg/dL"},
    "triglycerides":    {"normal": (0, 150),    "unit": "mg/dL"},
    "blood sugar":      {"normal": (70, 140),   "unit": "mg/dL"},
    "fasting glucose":  {"normal": (70, 100),   "unit": "mg/dL"},
    "hba1c":            {"normal": (4.0, 5.7),  "unit": "%"},
    "creatinine":       {"male": (0.7, 1.3),   "female": (0.6, 1.1), "unit": "mg/dL"},
    "urea":             {"normal": (7, 20),     "unit": "mg/dL"},
    "uric acid":        {"male": (3.4, 7.0),   "female": (2.4, 6.0), "unit": "mg/dL"},
    "sgpt":             {"normal": (7, 56),     "unit": "U/L"},
    "sgot":             {"normal": (10, 40),    "unit": "U/L"},
    "bilirubin":        {"normal": (0.1, 1.2),  "unit": "mg/dL"},
    "tsh":              {"normal": (0.4, 4.0),  "unit": "mIU/L"},
    "vitamin d":        {"normal": (30, 100),   "unit": "ng/mL"},
    "vitamin b12":      {"normal": (200, 900),  "unit": "pg/mL"},
    "iron":             {"male": (65, 175),     "female": (50, 170), "unit": "µg/dL"},
    "calcium":          {"normal": (8.5, 10.5), "unit": "mg/dL"},
    "sodium":           {"normal": (136, 145),  "unit": "mEq/L"},
    "potassium":        {"normal": (3.5, 5.0),  "unit": "mEq/L"},
    "esr":              {"male": (0, 22),       "female": (0, 29), "unit": "mm/hr"},
}


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY: unit normalization
# ═════════════════════════════════════════════════════════════════════════════
def normalize_unit(u: str) -> str:
    if not u:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", u.lower())


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY: balanced JSON block extraction
# ═════════════════════════════════════════════════════════════════════════════
def extract_json_block(content: str) -> str | None:
    start = -1
    open_char = None
    close_char = None

    for i, ch in enumerate(content):
        if ch == "{":
            start = i
            open_char, close_char = "{", "}"
            break
        elif ch == "[":
            start = i
            open_char, close_char = "[", "]"
            break

    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(content)):
        ch = content[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return content[start:i + 1]

    return None


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY: truncated JSON repair
# ═════════════════════════════════════════════════════════════════════════════
def _repair_truncated_json(content: str) -> dict | list | None:
    content = content.strip()
    if not content or content[0] not in "{[":
        return None

    for trim in range(0, min(len(content), 500), 1):
        candidate = content if trim == 0 else content[:-trim]
        if not candidate:
            break

        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False

        for ch in candidate:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                open_braces += 1
            elif ch == "}":
                open_braces -= 1
            elif ch == "[":
                open_brackets += 1
            elif ch == "]":
                open_brackets -= 1

        if in_string:
            continue

        if open_braces >= 0 and open_brackets >= 0:
            stripped = candidate.rstrip().rstrip(",").rstrip()
            closing = "]" * open_brackets + "}" * open_braces
            attempt = stripped + closing
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    return None


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITY: robust JSON parser (4 fallback strategies)
# ═════════════════════════════════════════════════════════════════════════════
def safe_json_parse(content: str) -> dict | list:
    content = content.strip()

    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Attempt 2: balanced bracket extraction
    json_block = extract_json_block(content)
    if json_block:
        try:
            return json.loads(json_block)
        except json.JSONDecodeError:
            pass

    # Attempt 3: regex fallback
    match = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Attempt 4: truncated JSON repair
    repaired = _repair_truncated_json(content)
    if repaired is not None:
        return repaired

    raise ValueError(f"Could not parse JSON from LLM response:\n{content[:500]}")


# ═════════════════════════════════════════════════════════════════════════════
#  MISTRAL CALLER WITH RETRY
# ═════════════════════════════════════════════════════════════════════════════
def call_mistral(
    prompt: str,
    model: str = "mistral-medium-latest",
    max_tokens: int = 4000,
    retries: int = 2,
) -> dict | list:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return safe_json_parse(content)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(1)
                continue
    raise ValueError(
        f"Mistral API failed after {retries} attempts. Last error: {last_error}"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  FILE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    image = image.convert("L")
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda x: 0 if x < 140 else 255)

    try:
        text = pytesseract.image_to_string(image, config="--psm 6")
        return text
    except TesseractNotFoundError:
        raise ValueError(
            "Tesseract OCR is not installed. Download from "
            "https://github.com/UB-Mannheim/tesseract/wiki, "
            "ensure tesseract.exe is in your PATH, then restart."
        )


def extract_text(file_bytes: bytes, content_type: str) -> str:
    """Unified extraction: routes to PDF or image extractor based on MIME type."""
    if content_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    elif content_type.startswith("image/"):
        return extract_text_from_image(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {content_type}")


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 0 — REPORT TYPE CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════
def report_type_agent(text: str) -> dict:
    prompt = (
        "You are a medical report classifier.\n\n"
        "Given the raw text from a health report, classify it into one of these types:\n"
        '- "lab": Blood tests, CBC, metabolic panels, lipid profiles, thyroid panels, etc.\n'
        '- "microbiology": Urine culture, blood culture, stool culture, sensitivity reports.\n'
        '- "radiology": X-ray, MRI, CT scan, ultrasound reports.\n'
        '- "pathology": Biopsy, histopathology, cytology reports.\n'
        '- "other": Anything else (prescriptions, discharge summaries, etc.)\n\n'
        "Return ONLY this JSON:\n"
        "{\n"
        '  "report_type": "<one of: lab, microbiology, radiology, pathology, other>",\n'
        '  "confidence": "<high, medium, low>",\n'
        '  "has_numeric_parameters": <true or false>,\n'
        '  "description": "<one-line description of what this report contains>"\n'
        "}\n\n"
        "Report text:\n" + text[:3000]
    )
    result = call_mistral(prompt, max_tokens=500)
    if not isinstance(result, dict):
        raise ValueError(f"Report type agent expected a dict, got: {type(result)}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 1 — EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
def extraction_agent(text: str) -> list[dict]:
    prompt = (
        "You are a medical lab report data extraction agent.\n\n"
        "Given the raw text below (from OCR or PDF extraction of a health report), "
        "extract EVERY measurable health parameter you can find.\n\n"
        "For each parameter, extract:\n"
        "- name: the test/parameter name exactly as written\n"
        "- value: the numeric or textual result\n"
        "- unit: the measurement unit (if present)\n"
        "- reference_range: the normal/reference range string (if present)\n\n"
        "Rules:\n"
        "- Extract ALL parameters, even if some fields are missing.\n"
        "- Do NOT interpret, classify, or judge values — just extract.\n"
        "- If a field is not present in the text, use null.\n"
        "- Do NOT hallucinate values not present in the text.\n"
        "- Preserve the original value exactly (do not round or convert).\n\n"
        "Return ONLY a JSON array. Example:\n"
        "[\n"
        '  {"name": "Hemoglobin", "value": "14.2", "unit": "g/dL", "reference_range": "13.5-17.5"},\n'
        '  {"name": "Blood Sugar (Fasting)", "value": "98", "unit": "mg/dL", "reference_range": "70-100"}\n'
        "]\n\n"
        "Raw report text:\n" + text[:8000]
    )
    result = call_mistral(prompt, max_tokens=6000)
    if not isinstance(result, list):
        raise ValueError(f"Extraction agent expected a JSON array, got: {type(result)}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 2 — STRUCTURING
# ═════════════════════════════════════════════════════════════════════════════
def structuring_agent(raw_params: list[dict]) -> list[dict]:
    prompt = (
        "You are a medical data structuring agent.\n\n"
        "Given the following raw extracted health parameters (JSON array), "
        "normalize and clean them into a consistent structure.\n\n"
        "For each parameter:\n"
        "1. Standardize the name (e.g., 'Hb' → 'Hemoglobin', 'FBS' → 'Fasting Blood Sugar', "
        "'TC' → 'Total Cholesterol', 'SGPT/ALT' → 'SGPT (ALT)').\n"
        "2. Ensure value is a clean numeric string where possible (remove spaces, commas in numbers).\n"
        "3. Standardize units (e.g., 'gm/dl' → 'g/dL', 'mg/dl' → 'mg/dL').\n"
        "4. Parse reference_range into ref_low and ref_high numeric values where possible.\n"
        "5. If reference_range is missing, set ref_low and ref_high to null.\n\n"
        "Return ONLY a JSON array with this exact structure per item:\n"
        "{\n"
        '  "name": "<standardized name>",\n'
        '  "original_name": "<name from input>",\n'
        '  "value": "<cleaned value string>",\n'
        '  "numeric_value": <float or null>,\n'
        '  "unit": "<standardized unit>",\n'
        '  "reference_range": "<original range string or null>",\n'
        '  "ref_low": <float or null>,\n'
        '  "ref_high": <float or null>\n'
        "}\n\n"
        "Rules:\n"
        "- Preserve all parameters, even those with missing fields.\n"
        "- If value cannot be parsed as a number, set numeric_value to null.\n"
        "- Do NOT add parameters that were not in the input.\n\n"
        "Raw parameters:\n" + json.dumps(raw_params, indent=2)
    )
    result = call_mistral(prompt, max_tokens=6000)
    if not isinstance(result, list):
        raise ValueError(f"Structuring agent expected a JSON array, got: {type(result)}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 3 — VALIDATION (hybrid)
# ═════════════════════════════════════════════════════════════════════════════
def _rule_based_validate(param: dict, gender: str = "male") -> dict | None:
    name_lower = param.get("name", "").lower()
    numeric_val = param.get("numeric_value")

    if numeric_val is None:
        return None

    for key, ranges in REFERENCE_RANGES.items():
        if key in name_lower:
            expected_unit = ranges.get("unit", "")
            param_unit = param.get("unit") or ""

            if normalize_unit(expected_unit) != normalize_unit(param_unit):
                if param_unit:
                    continue

            if "male" in ranges and "female" in ranges:
                low, high = ranges[gender]
            elif "normal" in ranges:
                low, high = ranges["normal"]
            else:
                continue

            if numeric_val < low:
                deviation = (low - numeric_val) / low if low != 0 else 0
                param["status"] = "Low"
                param["severity"] = "Moderate" if deviation > 0.15 else "Mild"
            elif numeric_val > high:
                deviation = (numeric_val - high) / high if high != 0 else 0
                param["status"] = "High"
                param["severity"] = "High" if deviation > 0.25 else "Moderate"
            else:
                param["status"] = "Normal"
                param["severity"] = "None"

            param["validation_source"] = "rule_based"
            param["validated_ref_range"] = f"{low}-{high}"
            return param

    return None


def _llm_validate(params_to_validate: list[dict]) -> list[dict]:
    if not params_to_validate:
        return []

    prompt = (
        "You are a medical lab result validation agent.\n\n"
        "For each parameter below, determine whether the value is Normal, Low, or High "
        "based on the provided reference range (ref_low / ref_high). "
        "If no reference range is available, use standard medical reference ranges.\n\n"
        "For each parameter, add these fields:\n"
        '- "status": "Normal" | "Low" | "High" | "Unknown"\n'
        '- "severity": "None" | "Mild" | "Moderate" | "High" | "Unknown"\n'
        '- "validation_source": "llm"\n'
        '- "validated_ref_range": "<low>-<high>" (the range you used)\n\n'
        "Severity guide:\n"
        "- Mild: slightly outside range (within ~10-15% deviation)\n"
        "- Moderate: moderately outside range (15-25% deviation)\n"
        "- High: significantly outside range (>25% deviation)\n\n"
        "Return the SAME JSON array with these fields added to each item.\n"
        "Do NOT remove any existing fields.\n\n"
        "Parameters to validate:\n" + json.dumps(params_to_validate, indent=2)
    )
    result = call_mistral(prompt, max_tokens=6000)
    if not isinstance(result, list):
        raise ValueError(f"LLM validation expected array, got: {type(result)}")
    return result


def validation_agent(structured_params: list[dict], gender: str = "male") -> list[dict]:
    validated = []
    needs_llm = []

    for param in structured_params:
        rule_result = _rule_based_validate(param.copy(), gender)
        if rule_result is not None:
            validated.append(rule_result)
        else:
            needs_llm.append(param)

    if needs_llm:
        llm_results = _llm_validate(needs_llm)
        validated.extend(llm_results)

    return validated


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 4 — RISK ANALYSIS (blended 80/20)
# ═════════════════════════════════════════════════════════════════════════════
def risk_agent(validated_params: list[dict]) -> dict:
    severity_points = {"Mild": 10, "Moderate": 20, "High": 30}
    rule_score = 0
    abnormal_params = []

    for param in validated_params:
        status = param.get("status", "")
        severity = param.get("severity", "")
        if status in ("Low", "High"):
            rule_score += severity_points.get(severity, 10)
            abnormal_params.append(param)

    rule_score = min(rule_score, 100)

    prompt = (
        "You are a medical risk analysis agent.\n\n"
        "Given the validated health parameters below, provide a clinical risk assessment.\n\n"
        "A rule-based system has already calculated a preliminary risk score of "
        f"{rule_score}/100 based on {len(abnormal_params)} abnormal parameters.\n\n"
        "Your job:\n"
        "1. Review ALL parameters for clinical patterns and concerning combinations.\n"
        "2. Provide your own risk score (0-100) with brief reasoning.\n\n"
        "Return ONLY this JSON:\n"
        "{\n"
        '  "llm_score": <integer 0-100>,\n'
        '  "score_reasoning": "<1-2 sentences ONLY — be concise>",\n'
        '  "clinical_patterns": ["<short pattern, max 15 words each>"],\n'
        '  "priority_parameters": ["<param name>"]\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "- score_reasoning MUST be 1-2 sentences maximum. No bullet points, no lists.\n"
        "- clinical_patterns: max 4 items, each max 15 words.\n"
        "- priority_parameters: max 3 items, just the parameter name.\n"
        "- Do NOT include detailed medical explanations or numbered sub-points.\n"
        "- Keep total response under 300 words.\n\n"
        "Risk category guide:\n"
        "- Low: 0-25 | Moderate: 26-50 | High: 51-75 | Critical: 76-100\n\n"
        "All validated parameters:\n" + json.dumps(validated_params, indent=2)
    )
    llm_result = call_mistral(prompt, max_tokens=2000)
    if not isinstance(llm_result, dict):
        raise ValueError(f"Risk agent expected a dict, got: {type(llm_result)}")

    llm_score = llm_result.get("llm_score", rule_score)
    if rule_score == 0:
        blended_score = int(0.5 * llm_score)
    else:
        blended_score = int(0.8 * rule_score + 0.2 * llm_score)
    blended_score = min(max(blended_score, 0), 100)

    if blended_score <= 25:
        category = "Low"
    elif blended_score <= 50:
        category = "Moderate"
    elif blended_score <= 75:
        category = "High"
    else:
        category = "Critical"

    return {
        "rule_based_score": rule_score,
        "llm_score": llm_score,
        "blended_score": blended_score,
        "risk_category": category,
        "score_reasoning": llm_result.get("score_reasoning", ""),
        "clinical_patterns": llm_result.get("clinical_patterns", []),
        "priority_parameters": llm_result.get("priority_parameters", []),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT 5 — EXPLANATION
# ═════════════════════════════════════════════════════════════════════════════
def explanation_agent(validated_params: list[dict], risk_assessment: dict) -> dict:
    prompt = (
        "You are a medical explanation agent. Your audience is the patient (non-medical person).\n\n"
        "Given the validated health parameters and risk assessment below, generate:\n"
        "1. A plain-language summary of the overall health picture.\n"
        "2. Key concerns explained in simple terms (what it means, why it matters).\n"
        "3. Actionable general advice (lifestyle, follow-up tests, when to see a doctor).\n\n"
        "Rules:\n"
        "- Use simple, non-technical language.\n"
        "- Do NOT make diagnoses — only flag areas of concern.\n"
        "- Be empathetic but factual.\n"
        "- Always recommend consulting a healthcare professional.\n"
        "- If everything is normal, say so clearly and positively.\n"
        "- summary: 2-3 sentences MAX.\n"
        "- key_concerns: max 5 items. Each explanation and why_it_matters: 1 sentence each.\n"
        "- general_advice: max 5 items, 1 sentence each.\n"
        "- follow_up_tests: max 3 items, just the test name.\n"
        "- Keep total response under 500 words.\n\n"
        "Return ONLY this JSON:\n"
        "{\n"
        '  "summary": "<2-3 sentence plain-language overview>",\n'
        '  "key_concerns": [\n'
        "    {\n"
        '      "parameter": "<name>",\n'
        '      "status": "<Low/High>",\n'
        '      "explanation": "<what this means in simple terms>",\n'
        '      "why_it_matters": "<potential health implications>"\n'
        "    }\n"
        "  ],\n"
        '  "general_advice": [\n'
        '    "<actionable advice item 1>",\n'
        '    "<actionable advice item 2>"\n'
        "  ],\n"
        '  "follow_up_tests": ["<recommended follow-up test if any>"],\n'
        '  "urgency": "Routine" | "Soon" | "Urgent"\n'
        "}\n\n"
        "Validated parameters:\n" + json.dumps(validated_params, indent=2) + "\n\n"
        "Risk assessment:\n" + json.dumps(risk_assessment, indent=2)
    )
    result = call_mistral(prompt, max_tokens=3000)
    if not isinstance(result, dict):
        raise ValueError(f"Explanation agent expected a dict, got: {type(result)}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  NON-LAB FALLBACK AGENT
# ═════════════════════════════════════════════════════════════════════════════
def non_lab_analysis_agent(text: str, report_type: str) -> dict:
    prompt = (
        f"You are a medical report analysis agent specialized in {report_type} reports.\n\n"
        "Analyze the following report and provide:\n"
        "1. A structured summary of findings.\n"
        "2. Key observations (normal and abnormal).\n"
        "3. Any recommended follow-up actions.\n\n"
        "Return ONLY this JSON:\n"
        "{\n"
        '  "report_type": "' + report_type + '",\n'
        '  "summary": "<2-3 sentence overview of the report>",\n'
        '  "findings": [\n'
        "    {\n"
        '      "finding": "<description>",\n'
        '      "status": "Normal" | "Abnormal" | "Inconclusive",\n'
        '      "significance": "<what this means>"\n'
        "    }\n"
        "  ],\n"
        '  "key_concerns": ["<concern if any>"],\n'
        '  "general_advice": ["<advice>"],\n'
        '  "urgency": "Routine" | "Soon" | "Urgent"\n'
        "}\n\n"
        "Report text:\n" + text
    )
    result = call_mistral(prompt)
    if not isinstance(result, dict):
        raise ValueError(f"Non-lab agent expected a dict, got: {type(result)}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    text: str,
    gender: str = "male",
    progress_callback=None,
) -> dict:
    results = {"agent_log": []}

    def _log(name, duration, **kwargs):
        entry = {"agent": name, "duration_seconds": round(duration, 2)}
        entry.update(kwargs)
        results["agent_log"].append(entry)

    def _progress(label, value):
        if progress_callback:
            progress_callback(label, value)

    # Agent 0: Classification
    _progress("Agent 0: Classifying report type...", 0.05)
    t0 = time.time()
    report_info = report_type_agent(text)
    _log("report_classification", time.time() - t0)
    results["report_info"] = report_info

    report_type = report_info.get("report_type", "other")
    has_numeric = report_info.get("has_numeric_parameters", False)

    # Non-lab path
    if report_type != "lab" and not has_numeric:
        _progress(f"Analyzing {report_type} report...", 0.5)
        t0 = time.time()
        non_lab = non_lab_analysis_agent(text, report_type)
        _log("non_lab_analysis", time.time() - t0)
        results["non_lab_analysis"] = non_lab
        results["pipeline_type"] = "non_lab"
        _progress("Analysis complete!", 1.0)
        return results

    # Lab pipeline
    results["pipeline_type"] = "lab"

    _progress("Agent 1: Extracting parameters...", 0.15)
    t0 = time.time()
    raw = extraction_agent(text)
    _log("extraction", time.time() - t0, output_count=len(raw))
    results["raw_extraction"] = raw

    _progress("Agent 2: Structuring data...", 0.35)
    t0 = time.time()
    structured = structuring_agent(raw)
    _log("structuring", time.time() - t0, input_count=len(raw), output_count=len(structured))
    results["structured_parameters"] = structured

    _progress("Agent 3: Validating values...", 0.55)
    t0 = time.time()
    validated = validation_agent(structured, gender)
    rule_count = sum(1 for p in validated if p.get("validation_source") == "rule_based")
    llm_count = sum(1 for p in validated if p.get("validation_source") == "llm")
    _log("validation", time.time() - t0, input_count=len(structured), output_count=len(validated))
    results["validated_parameters"] = validated
    results["validation_stats"] = {"rule_based": rule_count, "llm_fallback": llm_count}

    _progress("Agent 4: Analyzing risk...", 0.75)
    t0 = time.time()
    risk = risk_agent(validated)
    _log("risk_analysis", time.time() - t0)
    results["risk_assessment"] = risk

    _progress("Agent 5: Generating explanations...", 0.90)
    t0 = time.time()
    explanation = explanation_agent(validated, risk)
    _log("explanation", time.time() - t0)
    results["explanation"] = explanation

    # Summary
    normal = sum(1 for p in validated if p.get("status") == "Normal")
    abnormal = sum(1 for p in validated if p.get("status") in ("Low", "High"))
    unknown = sum(1 for p in validated if p.get("status") in ("Unknown", None, ""))
    results["summary"] = {
        "total_parameters": len(validated),
        "normal_count": normal,
        "abnormal_count": abnormal,
        "unknown_count": unknown,
        "risk_score": risk.get("blended_score", 0),
        "risk_category": risk.get("risk_category", "Unknown"),
    }

    _progress("Analysis complete!", 1.0)
    return results