import pytest
import os
from evaluation.coding_suite.isolated_executor import run_process_isolated_test, audit_code_safety


def test_adversarial_file_access_rejection():
    # Attempt to open file
    res = run_process_isolated_test("f = open('secret.txt', 'r')", "assert True")
    assert res["passed"] is False
    assert res["rejected_by_policy"] is True
    assert "SecurityRejection" in res["error"]


def test_adversarial_import_rejection():
    # Attempt direct import
    res1 = run_process_isolated_test("import os", "assert True")
    assert res1["passed"] is False
    assert res1["rejected_by_policy"] is True

    # Attempt from-import
    res2 = run_process_isolated_test("from subprocess import run", "assert True")
    assert res2["passed"] is False
    assert res2["rejected_by_policy"] is True

    # Attempt math import (all imports are prohibited in restricted runner)
    res3 = run_process_isolated_test("import math", "assert True")
    assert res3["passed"] is False
    assert res3["rejected_by_policy"] is True


def test_adversarial_dynamic_execution_rejection():
    # eval
    res_eval = run_process_isolated_test("eval('1+1')", "assert True")
    assert res_eval["passed"] is False
    assert res_eval["rejected_by_policy"] is True

    # exec
    res_exec = run_process_isolated_test("exec('x = 1')", "assert True")
    assert res_exec["passed"] is False
    assert res_exec["rejected_by_policy"] is True

    # compile
    res_compile = run_process_isolated_test("compile('x = 1', '', 'exec')", "assert True")
    assert res_compile["passed"] is False
    assert res_compile["rejected_by_policy"] is True


def test_adversarial_attribute_chain_hopping_rejection():
    # Attempting to access object.__subclasses__() via __class__
    res_sub = run_process_isolated_test("x = ().__class__.__bases__[0].__subclasses__()", "assert True")
    assert res_sub["passed"] is False
    assert res_sub["rejected_by_policy"] is True

    # Attempting to access __globals__
    res_glob = run_process_isolated_test("def f(): pass\ng = f.__globals__", "assert True")
    assert res_glob["passed"] is False
    assert res_glob["rejected_by_policy"] is True


def test_adversarial_fixture_rejection():
    # Both code AND test fixtures must be audited
    res = run_process_isolated_test("def add(a, b): return a + b", "import os; assert add(2, 3) == 5")
    assert res["passed"] is False
    assert res["rejected_by_policy"] is True
    assert "FixtureSecurityRejection" in res["error"]


def test_adversarial_infinite_loop_timeout():
    # Infinite loop must cleanly terminate within timeout
    res = run_process_isolated_test("def loop():\n    while True: pass\nloop()", "assert True", timeout_seconds=1.0)
    assert res["passed"] is False
    assert res["timed_out"] is True
    assert "TimeoutError" in res["error"]


def test_oversized_code_rejection():
    # Code exceeding size limit
    big_code = "# " + ("x" * (40 * 1024))
    res = run_process_isolated_test(big_code, "assert True")
    assert res["passed"] is False
    assert res["rejected_by_policy"] is True
