"""
dashboard/app.py

PotholeOps Web Dashboard Launcher:
  1. Checks if the FastAPI backend service is running on http://localhost:8000
  2. Starts the FastAPI uvicorn server if not already running
  3. Opens http://localhost:8000/ directly in your web browser

Run: python dashboard/app.py
"""

import os
import sys
import time
import webbrowser
import requests
import uvicorn

API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_running() -> bool:
    """Check if FastAPI service is online."""
    try:
        res = requests.get(f"{API_URL}/health", timeout=2)
        return res.status_code == 200
    except Exception:
        return False


def main():
    print(f"PotholeOps Modern Web Dashboard Launcher")
    print(f"Connecting to FastAPI backend at {API_URL}...")

    if not check_api_running():
        print("FastAPI server is not running. Starting uvicorn server...")
        # Open web browser after a brief delay
        def launch_browser():
            time.sleep(1.5)
            print(f"Opening dashboard in browser at {API_URL}...")
            webbrowser.open(API_URL)

        import threading
        threading.Thread(target=launch_browser, daemon=True).start()

        # Run uvicorn server
        uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print(f"FastAPI server is already running! Opening {API_URL} in web browser...")
        webbrowser.open(API_URL)


if __name__ == "__main__":
    main()
