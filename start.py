import os
import subprocess
import sys

port = os.environ.get("PORT", "8501")
subprocess.run([
    "streamlit", "run", "nvidia_rag_app.py",
    "--server.port", port,
    "--server.address", "0.0.0.0",
    "--server.headless", "true"
])
