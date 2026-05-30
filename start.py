import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    dirs = ["data", "logs"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    print("✅ Environment setup complete")

def run_streamlit():
    port = os.environ.get("PORT", "8501")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "nvidia_rag_app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    print(f"🚀 Starting on port {port}")
    subprocess.run(cmd)

def main():
    print("🧠 DocuMentor - Starting on Railway")
    setup_environment()
    run_streamlit()

if __name__ == "__main__":
    main()
