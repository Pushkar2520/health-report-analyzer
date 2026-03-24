# AI-Powered Health Report Analyzer

This Python project provides a production-ready AI-powered health report analyzer with a user-friendly Streamlit interface. It extracts text from PDF or image health reports, uses OpenAI API for initial analysis, applies rule-based validation, calculates a risk score, and displays results in a clean UI.

## Features

- **File Upload**: Support for PDF and image files (PNG, JPG, JPEG)
- **Text Extraction**: Uses pdfplumber for PDFs and pytesseract for images
- **AI Analysis**: Leverages OpenAI GPT-4o-mini to extract and classify health parameters
- **Rule-Based Validation**: Applies predefined reference ranges for accuracy
- **Risk Scoring**: Generates a 0-100 risk score based on abnormal values
- **Streamlit UI**: Clean interface for uploading, analyzing, and viewing results
- **JSON Download**: Allows users to download structured analysis results

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate   # Windows
   # or `source venv/bin/activate` on macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set the `OPENAI_API_KEY` environment variable with your API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. For image processing, install Tesseract OCR:
   - **Windows**: Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Usage

### Streamlit Web App
Run the interactive web application:
```bash
streamlit run app.py
```

This will start a local web server. Open the provided URL in your browser to use the app.

### Command Line (Original Script)
For command-line usage:
```bash
python health_report_analyzer.py path/to/health_report.pdf
```

## How It Works

1. **Upload**: User uploads a PDF or image file via the Streamlit interface
2. **Text Extraction**: 
   - PDFs: Extracted using pdfplumber
   - Images: OCR performed with pytesseract
3. **AI Analysis**: Text sent to OpenAI with a detailed prompt to extract health parameters
4. **Validation**: Rule-based checks against standard reference ranges
5. **Risk Calculation**: Score computed based on abnormal parameters and severity
6. **Display**: Results shown in the UI with option to download JSON

## Output Format

The analysis produces a structured JSON with:
- **Summary**: Total parameters, normal/abnormal counts
- **Parameters**: Detailed list with name, value, unit, reference range, status, severity, explanation
- **Key Concerns**: List of potential health issues
- **General Advice**: Recommendations
- **Risk Score**: 0-100 score indicating overall health risk

## Customization

- Modify reference ranges in `app.py` for different demographics
- Adjust the OpenAI prompt for different analysis focuses
- Extend validation rules for additional parameters

> **Important**: This tool is for informational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical decisions.
