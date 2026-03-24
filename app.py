"""
app.py
──────
Streamlit frontend for the Health Report Analyzer.
Sends files to the FastAPI backend (api.py) and renders results.

Run with:
    streamlit run app.py

Requires api.py to be running:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import streamlit as st
import requests
import requests.exceptions
import json
import os
import time
import threading
import pandas as pd

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# FIX #4: env-based API URL — deployable without code changes
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# FIX #3: max upload size (10 MB)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ═════════════════════════════════════════════════════════════════════════════
#  FIX #5: RETRY WRAPPER
# ═════════════════════════════════════════════════════════════════════════════
def call_with_retry(func, retries: int = 2, label: str = "API call"):
    """
    Retry a callable up to `retries` times.
    Distinguishes timeout errors from other failures for clear UX messaging.
    Raises:
        requests.exceptions.Timeout  — if all retries timed out
        ValueError                   — if all retries failed with other errors
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < retries:
                time.sleep(1)
                continue
            # All retries exhausted with timeout — re-raise as Timeout
            raise
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(1)
                continue
    raise ValueError(f"{label} failed after {retries} attempts. Last error: {last_error}")


# ═════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════
def api_health_check() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  API CALLERS (with safe error parsing)
# ═════════════════════════════════════════════════════════════════════════════
def _parse_error(resp) -> str:
    """Safely extract error detail from a response, even if body isn't JSON."""
    try:
        return resp.json().get("detail", resp.text)
    except (ValueError, KeyError):
        return resp.text


def call_extract_text(file_bytes: bytes, file_name: str, content_type: str) -> dict:
    """POST to /extract-text."""
    resp = requests.post(
        f"{API_BASE_URL}/extract-text",
        files={"file": (file_name, file_bytes, content_type)},
        timeout=60,
    )
    if resp.status_code != 200:
        raise ValueError(f"Text extraction failed ({resp.status_code}): {_parse_error(resp)}")
    return resp.json()


def call_analyze(file_bytes: bytes, file_name: str, content_type: str, gender: str) -> dict:
    """POST to /analyze."""
    resp = requests.post(
        f"{API_BASE_URL}/analyze",
        files={"file": (file_name, file_bytes, content_type)},
        data={"gender": gender},
        timeout=300,
    )
    if resp.status_code != 200:
        raise ValueError(f"Analysis failed ({resp.status_code}): {_parse_error(resp)}")
    return resp.json()


def call_analyze_text(text: str, gender: str) -> dict:
    """POST to /analyze-text."""
    resp = requests.post(
        f"{API_BASE_URL}/analyze-text",
        json={"text": text, "gender": gender},
        timeout=300,
    )
    if resp.status_code != 200:
        raise ValueError(f"Analysis failed ({resp.status_code}): {_parse_error(resp)}")
    return resp.json()


# ═════════════════════════════════════════════════════════════════════════════
#  FIX #2: PROGRESS SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
def _run_analysis_with_progress(api_call_func):
    """
    Run an API call in a background thread while showing live progress.
    The progress bar animates through agent steps while the blocking API call
    runs in a separate thread, giving real-time feedback.
    Returns the results dict.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        (0.05, "🏷️ Agent 0: Classifying report type..."),
        (0.15, "📄 Agent 1: Extracting parameters..."),
        (0.35, "🔧 Agent 2: Structuring data..."),
        (0.55, "✅ Agent 3: Validating values..."),
        (0.75, "⚠️ Agent 4: Analyzing risk..."),
        (0.90, "💬 Agent 5: Generating explanations..."),
    ]

    # Phase 1: animate pre-API steps quickly to show activity
    for pct, label in steps[:3]:  # agents 0, 1, 2
        status_text.text(label)
        progress_bar.progress(pct)
        time.sleep(0.3)

    # Phase 2: run API call in background thread while animating remaining steps
    result_container = {"data": None, "error": None}

    def _run_call():
        try:
            result_container["data"] = api_call_func()
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=_run_call)
    thread.start()

    # Animate steps[3:] while the thread is alive
    for pct, label in steps[3:]:
        if not thread.is_alive():
            break
        status_text.text(label)
        progress_bar.progress(pct)
        time.sleep(2.0)  # slower pace — real API takes ~10-30s per agent

    # If API is still running after all steps animated, hold at 90%
    while thread.is_alive():
        status_text.text("💬 Agent 5: Generating explanations...")
        progress_bar.progress(0.90)
        time.sleep(1.0)

    thread.join()

    # Check for errors from the background thread
    if result_container["error"] is not None:
        progress_bar.empty()
        status_text.empty()
        raise result_container["error"]

    results = result_container["data"]

    # Phase 3: completion
    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)

    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  UI RENDERERS
# ═════════════════════════════════════════════════════════════════════════════
def _render_non_lab_results(results: dict):
    """Render results for non-lab reports."""
    report_info = results.get("report_info", {})
    analysis = results.get("non_lab_analysis", {})

    st.markdown("---")
    st.subheader(f"📋 {report_info.get('report_type', 'Non-Lab').title()} Report Analysis")
    st.caption(report_info.get("description", ""))

    st.info(analysis.get("summary", "No summary available."))

    urgency = analysis.get("urgency", "Routine")
    if urgency == "Urgent":
        st.error(f"🔴 Urgency: {urgency} — Please consult a doctor promptly.")
    elif urgency == "Soon":
        st.warning(f"🟡 Urgency: {urgency} — Consider scheduling a follow-up.")
    else:
        st.success(f"🟢 Urgency: {urgency}")

    findings = analysis.get("findings", [])
    if findings:
        st.subheader("🔬 Findings")
        for f in findings:
            status = f.get("status", "")
            icon = "✅" if status == "Normal" else "⚠️" if status == "Abnormal" else "❓"
            with st.expander(f"{icon} {f.get('finding', 'Unknown')}"):
                st.write(f"**Status:** {status}")
                st.write(f"**Significance:** {f.get('significance', 'N/A')}")

    concerns = analysis.get("key_concerns", [])
    if concerns:
        st.subheader("⚠️ Key Concerns")
        for c in concerns:
            st.write(f"• {c}")

    advice = analysis.get("general_advice", [])
    if advice:
        st.subheader("💡 Advice")
        for a in advice:
            st.write(f"• {a}")

    _render_agent_log(results)
    _render_download(results, "health_analysis_non_lab.json")


def _render_lab_results(results: dict):
    """Render results for lab reports."""
    validated = results["validated_parameters"]
    risk = results["risk_assessment"]
    explanation = results["explanation"]
    report_info = results.get("report_info", {})
    validation_stats = results.get("validation_stats", {})
    summary = results["summary"]

    st.markdown("---")

    st.caption(
        f"Report type: **{report_info.get('report_type', 'lab')}** "
        f"(confidence: {report_info.get('confidence', 'N/A')})"
    )

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parameters Found", summary["total_parameters"])
    col2.metric("Normal", summary["normal_count"])
    col3.metric("Abnormal", summary["abnormal_count"])
    col4.metric("Risk Score", f"{summary['risk_score']}/100", delta=summary["risk_category"])

    st.caption(
        f"Score breakdown: rule-based {risk.get('rule_based_score', '?')}, "
        f"LLM {risk.get('llm_score', '?')}, "
        f"blended (80/20) → **{risk.get('blended_score', '?')}**"
    )

    # Summary
    st.subheader("💬 Summary")
    st.info(explanation.get("summary", "No summary available."))

    urgency = explanation.get("urgency", "Routine")
    if urgency == "Urgent":
        st.error(f"🔴 Urgency: {urgency} — Please consult a doctor promptly.")
    elif urgency == "Soon":
        st.warning(f"🟡 Urgency: {urgency} — Consider scheduling a follow-up.")
    else:
        st.success(f"🟢 Urgency: {urgency}")

    # Concerns
    concerns = explanation.get("key_concerns", [])
    if concerns:
        st.subheader("⚠️ Key Concerns")
        for c in concerns:
            icon = "🔴" if c.get("status") == "High" else "🟡"
            with st.expander(f"{icon} {c.get('parameter', 'Unknown')} — {c.get('status', '')}"):
                st.write(f"**What it means:** {c.get('explanation', 'N/A')}")
                st.write(f"**Why it matters:** {c.get('why_it_matters', 'N/A')}")

    # Advice
    advice = explanation.get("general_advice", [])
    if advice:
        st.subheader("💡 General Advice")
        for a in advice:
            st.write(f"• {a}")

    follow_up = explanation.get("follow_up_tests", [])
    if follow_up:
        st.subheader("🔬 Recommended Follow-up Tests")
        for f in follow_up:
            st.write(f"• {f}")

    # Clinical patterns
    patterns = risk.get("clinical_patterns", [])
    if patterns:
        st.subheader("🔍 Clinical Patterns Detected")
        for p in patterns:
            st.write(f"• {p}")

    # FIX #6: Parameter table — sorted by classification (abnormal first)
    st.subheader("📊 Detailed Parameters")
    param_display = []
    # Sort order: High severity first, then Moderate, Mild, Normal, Unknown last
    severity_order = {"High": 0, "Moderate": 1, "Mild": 2, "None": 3, "Unknown": 4, "N/A": 5}

    for p in validated:
        status = p.get("status", "Unknown")
        severity = p.get("severity", "N/A")
        if status == "Normal":
            icon = "✅"
        elif status in ("Low", "High"):
            icon = "⚠️" if severity in ("Mild", "Moderate") else "🔴"
        else:
            icon = "❓"
        param_display.append({
            "Status": icon,
            "Parameter": p.get("name", "Unknown"),
            "Value": p.get("value", "N/A"),
            "Unit": p.get("unit", ""),
            "Reference Range": p.get("validated_ref_range", p.get("reference_range", "N/A")),
            "Classification": f"{status} ({severity})",
            "Source": p.get("validation_source", "N/A"),
            "_sort_key": severity_order.get(severity, 5),
        })

    df = pd.DataFrame(param_display)
    df = df.sort_values(by="_sort_key", ascending=True).drop(columns=["_sort_key"])
    df = df.reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

    st.caption(
        f"Validation: {validation_stats.get('rule_based', 0)} rule-based, "
        f"{validation_stats.get('llm_fallback', 0)} LLM-validated"
    )

    _render_agent_log(results)

    with st.expander("🔧 Raw Agent Outputs (Debug)"):
        st.json(results)

    _render_download(results, "health_analysis_multi_agent.json")


def _render_agent_log(results: dict):
    """Render agent timing log."""
    agent_log = results.get("agent_log", [])
    if agent_log:
        with st.expander("⏱️ Agent Performance Log"):
            for entry in agent_log:
                line = f"**{entry['agent']}**: {entry['duration_seconds']}s"
                if "output_count" in entry:
                    line += f" ({entry['output_count']} items)"
                st.write(line)


def _render_download(results: dict, filename: str):
    """Render JSON download button."""
    json_str = json.dumps(results, indent=2)
    st.download_button(
        label="📥 Download Full Analysis (JSON)",
        data=json_str,
        file_name=filename,
        mime="application/json",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  FIX #1: UNIFIED ANALYSIS RUNNER (timeout handling + progress + retry)
# ═════════════════════════════════════════════════════════════════════════════
def _run_and_render(api_call_func, label: str = "analysis"):
    """
    Unified handler for both file upload and text analysis:
    - Shows simulated progress (FIX #2)
    - Retries on transient failure (FIX #5)
    - Shows specific timeout message (FIX #1)
    - Routes to correct renderer
    """
    try:
        results = _run_analysis_with_progress(
            lambda: call_with_retry(api_call_func, retries=2, label=label)
        )
        _route_results(results)
    except requests.exceptions.Timeout:
        st.error(
            "⏳ **Request timed out.** The backend took too long to respond.\n\n"
            "Try again, or upload a smaller file. If this keeps happening, "
            "the Mistral API may be experiencing delays."
        )
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Health Report Analyzer", page_icon="🩺", layout="wide")
    st.title("🩺 AI-Powered Health Report Analyzer")
    st.warning(
        "⚠️ This is NOT a medical diagnosis tool. "
        "Always consult a qualified healthcare professional for medical advice."
    )

    # ── Check API connection ──
    if not api_health_check():
        st.error(
            "❌ Cannot connect to the FastAPI backend at "
            f"**{API_BASE_URL}**.\n\n"
            "Start the backend first:\n"
            "```\nuvicorn api:app --host 0.0.0.0 --port 8000 --reload\n```"
        )
        st.stop()

    st.success(f"✅ Connected to API at {API_BASE_URL}")

    # ── Sidebar ──
    with st.sidebar:
        st.header("Settings")
        gender = st.selectbox("Gender (for reference ranges)", ["male", "female"], index=0)
        st.markdown("---")
        st.markdown(
            "**Pipeline Agents:**\n"
            "0. 🏷️ Report Classifier\n"
            "1. 📄 Extraction\n"
            "2. 🔧 Structuring\n"
            "3. ✅ Validation (hybrid)\n"
            "4. ⚠️ Risk Analysis (blended)\n"
            "5. 💬 Explanation"
        )
        st.markdown("---")
        st.caption("Risk scoring: 80% rule-based + 20% LLM")
        st.markdown("---")
        st.caption(f"API: {API_BASE_URL}")

    # ── Input tabs ──
    tab_upload, tab_text = st.tabs(["📁 Upload File", "📝 Paste Text"])

    # ── Tab 1: File Upload ──
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a PDF or image of your health report",
            type=["pdf", "png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_name = uploaded_file.name
            content_type = uploaded_file.type

            # FIX #3: reject files over 10 MB
            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                st.error(
                    f"📁 File too large ({len(file_bytes) / (1024*1024):.1f} MB). "
                    f"Maximum allowed size: {MAX_FILE_SIZE_MB} MB."
                )
                st.stop()

            # Cache key to avoid re-extracting on every rerun
            cache_key = f"extracted_{file_name}_{len(file_bytes)}"

            # Preview extracted text (cached in session_state)
            with st.expander("📄 Preview Extracted Text", expanded=False):
                if cache_key not in st.session_state:
                    try:
                        extract_resp = call_extract_text(file_bytes, file_name, content_type)
                        st.session_state[cache_key] = extract_resp["text"]
                    except ValueError as e:
                        st.session_state[cache_key] = None
                        st.error(str(e))

                cached_text = st.session_state.get(cache_key)
                if cached_text:
                    st.text_area("Extracted text", cached_text, height=200)

            # Analyze button
            if st.button("🚀 Analyze Report", type="primary", key="btn_upload"):
                _run_and_render(
                    lambda: call_analyze(file_bytes, file_name, content_type, gender),
                    label="File analysis",
                )

    # ── Tab 2: Paste Text ──
    with tab_text:
        pasted_text = st.text_area(
            "Paste your health report text here",
            height=250,
            placeholder="Paste the content of your lab report here...",
        )

        if st.button("🚀 Analyze Text", type="primary", key="btn_text"):
            if not pasted_text.strip():
                st.error("Please paste some text before analyzing.")
            else:
                _run_and_render(
                    lambda: call_analyze_text(pasted_text, gender),
                    label="Text analysis",
                )


def _route_results(results: dict):
    """Route to the correct renderer based on pipeline type."""
    pipeline_type = results.get("pipeline_type", "lab")
    if pipeline_type == "non_lab":
        _render_non_lab_results(results)
    else:
        _render_lab_results(results)


if __name__ == "__main__":
    main()