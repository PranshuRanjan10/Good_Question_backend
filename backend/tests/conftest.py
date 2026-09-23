"""Point the app at a throwaway database before anything imports app.db.session (the engine is
built at import time from IRONSENSE_DB_PATH), so tests never touch backend/ironsense.db."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("IRONSENSE_DB_PATH", str(Path(tempfile.mkdtemp(prefix="ironsense_test_")) / "test.db"))
