"""
Setup script for Step 113 Multi-Agent Benchmark Repository Fixture.

Creates a multi-file repository suitable for multi-agent coordination:
- app/models.py
- app/validation.py
- app/service.py
- app/security.py
- tests/test_validation.py
- tests/test_service.py
- tests/test_security.py
"""

from pathlib import Path
import shutil
import subprocess
import sys

def setup_step113_repository(bench_root: Path) -> Path:
    bench_root = Path(bench_root).resolve()
    if bench_root.exists():
        shutil.rmtree(bench_root)
    bench_root.mkdir(parents=True, exist_ok=True)

    repo_dir = bench_root / "repository"
    app_dir = repo_dir / "app"
    tests_dir = repo_dir / "tests"
    app_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")

    # models.py
    (app_dir / "models.py").write_text("""class UserAccount:
    def __init__(self, username: str, email: str, password_hash: str):
        self.username = username
        self.email = email
        self.password_hash = password_hash
""", encoding="utf-8")

    # security.py - initial stub (to be implemented by Implementer)
    (app_dir / "security.py").write_text("""# Security and password strength validation module
def validate_password_strength(password: str) -> bool:
    # Initial permissive stub (needs hardened password validation)
    return len(password) > 0
""", encoding="utf-8")

    # validation.py
    (app_dir / "validation.py").write_text("""from app.security import validate_password_strength

def validate_account_data(data: dict) -> bool:
    if not data.get("username") or not data.get("email"):
        return False
    pwd = data.get("password", "")
    if not validate_password_strength(pwd):
        return False
    return True
""", encoding="utf-8")

    # service.py
    (app_dir / "service.py").write_text("""import hashlib
from app.models import UserAccount
from app.validation import validate_account_data

class AccountService:
    def __init__(self):
        self.accounts = {}

    def register(self, data: dict) -> UserAccount:
        if not validate_account_data(data):
            raise ValueError("Registration validation failed")
        pwd_hash = hashlib.sha256(data["password"].encode("utf-8")).hexdigest()
        account = UserAccount(data["username"], data["email"], pwd_hash)
        self.accounts[account.username] = account
        return account
""", encoding="utf-8")

    # tests/test_validation.py
    (tests_dir / "test_validation.py").write_text("""from app.validation import validate_account_data

def test_validate_account_basic():
    assert validate_account_data({"username": "user1", "email": "u1@test.com", "password": "any"}) is True
    assert validate_account_data({"username": ""}) is False
""", encoding="utf-8")

    # tests/test_service.py
    (tests_dir / "test_service.py").write_text("""import pytest
from app.service import AccountService

def test_service_register_success():
    svc = AccountService()
    acc = svc.register({"username": "user1", "email": "u1@test.com", "password": "any"})
    assert acc.username == "user1"
    assert acc.email == "u1@test.com"

def test_service_register_invalid_rejected():
    svc = AccountService()
    with pytest.raises(ValueError):
        svc.register({"username": ""})
""", encoding="utf-8")

    # tests/test_security.py (initial passing stub test)
    (tests_dir / "test_security.py").write_text("""from app.security import validate_password_strength

def test_initial_stub():
    assert validate_password_strength("p") is True
""", encoding="utf-8")

    # Verify initial tests pass
    res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=str(repo_dir), capture_output=True, text=True)
    assert res.returncode == 0, f"Initial fixture test failure: {res.stderr}\n{res.stdout}"
    return repo_dir

if __name__ == "__main__":
    p = setup_step113_repository(Path("artifacts/step113_multi_agent_benchmark"))
    print("Step 113 repository fixture created at:", p)
