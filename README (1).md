# Medicine Batch

This project is a FastAPI web app that lets you upload a photo of a medicine strip or box, extract text using OCR, and check whether the medicine may be banned or restricted.

## Prerequisites for Windows

Before running the app, make sure you have:

- Python 3.10 or newer
- Git for Windows
- Tesseract OCR installed and available in your system PATH
- A Gemini API key from Google AI Studio

## Install Tesseract OCR on Windows

You can install Tesseract OCR on Windows using winget:

```powershell
winget install --id UB-Mannheim.TesseractOCR
```

By default, Tesseract is usually installed in:

```text
C:\Program Files\Tesseract-OCR\
```

After the installation finishes:

1. Close and reopen PowerShell.
2. Verify the installation:

```powershell
tesseract --version
```

If the command still does not work, you can check the installed location directly:

```powershell
Get-Command tesseract
```

If needed, you can also test it using the full path:

```powershell
C:\Program Files\Tesseract-OCR\tesseract.exe --version
```

## Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/raptor2307/Medicine-Batch.git
cd Medicine-Batch  (ithu miss akkal)
git fetch
git checkout development
git pull
```

## Set Up the Project

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

## Configure Environment Variables

Create a file named .env in the project root and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

## Run the App

Start the FastAPI server:

```powershell
uvicorn main:app --reload
```

Then open your browser at:

```text
http://localhost:8000
```

## How to Use the App

1. Open the homepage in your browser.
2. Upload a clear image of the medicine strip or box.
3. Submit the image.
4. The app will process the OCR text and show the analysis result.

## Troubleshooting

- If OCR does not work, make sure Tesseract is installed and available in PATH.
- If the app does not start, ensure all Python dependencies were installed successfully.
- If you see an API error, verify that .env contains a valid GEMINI_API_KEY.

