"""Production entry point: waitress WSGI server (works on Windows, Linux and in Docker).

    set HITDS_USERS=analyst:choose-a-password   (PowerShell: $env:HITDS_USERS="analyst:choose-a-password")
    python serve.py

Reads PORT (default 5000; Hugging Face Spaces uses 7860) and the HITDS_* variables documented in app/server.py.
"""
import os

from app.server import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    try:
        from waitress import serve

        print(f"HITDS serving on http://{host}:{port}  (waitress, 8 threads)")
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        print("waitress not installed - falling back to the Flask development server")
        app.run(host=host, port=port, threaded=True)
