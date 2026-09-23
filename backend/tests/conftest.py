"""Point the app at a throwaway database before anything imports app.db.session (the engine is
built at import time from IRONSENSE_DB_PATH), so tests never touch backend/ironsense.db."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="ironsense_test_"))
os.environ.setdefault("IRONSENSE_DB_PATH", str(_tmp / "test.db"))
os.environ.setdefault("IRONSENSE_LOG_DIR", str(_tmp / "raw"))     # raw telemetry logs, not backend/logs
