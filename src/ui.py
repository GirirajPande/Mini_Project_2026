"""
VisionGuard - Smart Surveillance System UI
Uses pywebview to render a modern HTML/CSS dashboard instead of basic Tkinter.
"""

import webview
import subprocess
import queue
import time
import os
import sys
from datetime import datetime

camera_process = None
log_queue = queue.Queue()


def get_html():
    html_path = os.path.join(os.path.dirname(__file__), "ui.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ui.html not found</h1>"


class SurveillanceAPI:
    def start_monitoring(self):
        global camera_process
        if camera_process is None or camera_process.poll() is not None:
            try:
                camera_process = subprocess.Popen(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "main.py")],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return {"status": "started", "message": "Monitoring started"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "already_running", "message": "Monitoring already active"}

    def stop_monitoring(self):
        global camera_process
        if camera_process is not None and camera_process.poll() is None:
            camera_process.terminate()
            camera_process = None
            return {"status": "stopped", "message": "Monitoring stopped"}
        return {"status": "not_running", "message": "Monitoring was not active"}

    def train_model(self):
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(__file__), "train_lbph.py")],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return {"status": "success", "message": result.stdout.strip()}
            return {"status": "error", "message": result.stderr.strip()}
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Training timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def test_telegram(self):
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from alert import send_telegram_alert
            send_telegram_alert("🔔 VisionGuard Test Alert - System is online")
            return {"status": "sent", "message": "Test alert sent to Telegram"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_status(self):
        global camera_process
        is_running = camera_process is not None and camera_process.poll() is None
        return {
            "monitoring": is_running,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "uptime": int(time.time())
        }


def main():
    api = SurveillanceAPI()

    webview.create_window(
        title="VisionGuard — Smart Surveillance System",
        html=get_html(),
        js_api=api,
        width=1200,
        height=750,
        min_size=(900, 600),
        background_color="#080c0a",
    )

    webview.start(debug=False)


if __name__ == "__main__":
    main()