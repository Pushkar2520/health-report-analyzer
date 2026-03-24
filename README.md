# Health Report Analyzer

AI-powered multi-agent system for analyzing medical health reports from PDF, image, or raw text inputs.  
The system extracts data, validates parameters using hybrid logic (rule-based + LLM), computes a risk score, and generates patient-friendly explanations.

---

## Overview

This project consists of a modular architecture with:

- FastAPI backend for processing and analysis
- Streamlit frontend for user interaction
- Multi-agent pipeline for structured medical analysis
- OCR and PDF parsing for text extraction
- Hybrid validation using medical reference ranges and LLM fallback
- Risk scoring and explanation generation

---

## Features

- Upload PDF or image-based health reports
- Extract text using OCR (pytesseract) and PDF parsing (pdfplumber)
- Analyze reports using a multi-agent pipeline:
  - Report classification
  - Parameter extraction
  - Data structuring
  - Validation (rule-based + LLM)
  - Risk analysis
  - Explanation generation
- Support for direct text input (no file upload required)
- Risk score (0–100) with severity classification
- Structured JSON output
- Interactive UI with progress simulation
- Download full analysis results

---

## Project Structure
Health_prediction/
│
├── api.py # FastAPI backend
├── app.py # Streamlit frontend
├── health_report_analyzer.py # Core multi-agent pipeline
├── requirements.txt
├── README.md
├── .gitignore
