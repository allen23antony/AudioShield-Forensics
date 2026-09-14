"""Entry-point for the AudioShield Forensics web server.

Run directly:
    python -m src.api.run

Or via the convenience batch file:
    run_web.bat
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
