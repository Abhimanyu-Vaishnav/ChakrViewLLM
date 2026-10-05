"""
Script to create the realistic on-disk benchmark repository fixture.
"""
import shutil
import subprocess
import sys
from pathlib import Path

bench_root = Path("artifacts/step112_autonomous_benchmark")
if bench_root.exists():
    shutil.rmtree(bench_root)
bench_root.mkdir(parents=True, exist_ok=True)
repo_dir = bench_root / "repository"
repo_dir.mkdir(parents=True, exist_ok=True)

# Directory structure
app_dir = repo_dir / "app"
tests_dir = repo_dir / "tests"
app_dir.mkdir(parents=True, exist_ok=True)
tests_dir.mkdir(parents=True, exist_ok=True)

(app_dir / "__init__.py").write_text("", encoding="utf-8")
(tests_dir / "__init__.py").write_text("", encoding="utf-8")

models_code = """class UserProfile:
    def __init__(self, username: str, email: str, age: int):
        self.username = username
        self.email = email
        self.age = age

    def to_dict(self):
        return {"username": self.username, "email": self.email, "age": self.age}
"""
(app_dir / "models.py").write_text(models_code, encoding="utf-8")

validation_code = """def validate_registration(data: dict) -> bool:
    # Existing incomplete validation (does not check age bounds)
    if not data.get("username") or not data.get("email"):
        return False
    return True
"""
(app_dir / "validation.py").write_text(validation_code, encoding="utf-8")

service_code = """from app.models import UserProfile
from app.validation import validate_registration

class RegistrationService:
    def __init__(self):
        self.users = {}

    def register(self, data: dict) -> UserProfile:
        if not validate_registration(data):
            raise ValueError("Invalid registration data")
        user = UserProfile(data["username"], data["email"], data.get("age", 0))
        self.users[user.username] = user
        return user
"""
(app_dir / "service.py").write_text(service_code, encoding="utf-8")

test_service_code = """import pytest
from app.service import RegistrationService

def test_registration_valid():
    svc = RegistrationService()
    user = svc.register({"username": "alice", "email": "alice@example.com", "age": 25})
    assert user.username == "alice"
    assert user.age == 25

def test_registration_missing_fields():
    svc = RegistrationService()
    with pytest.raises(ValueError):
        svc.register({"username": ""})
"""
(tests_dir / "test_service.py").write_text(test_service_code, encoding="utf-8")

test_validation_code = """from app.validation import validate_registration

def test_validate_registration_basic():
    assert validate_registration({"username": "bob", "email": "bob@example.com"}) is True
    assert validate_registration({"username": ""}) is False
"""
(tests_dir / "test_validation.py").write_text(test_validation_code, encoding="utf-8")

# Verify initial tests pass
res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=str(repo_dir), capture_output=True, text=True)
assert res.returncode == 0, f"Initial tests failed: {res.stderr}\n{res.stdout}"
print("Initial benchmark repository successfully created and verified!")
