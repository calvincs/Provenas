"""Research-phase tests (Phases 1-5): the calculator distillation, encoders, losses, the VMs, and the
typed/rewrite generators. These need torch + numpy and are skipped if those aren't installed.

    pip install -e ".[research]" && pytest tests/test_research.py
"""
import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from tests import sanity   # noqa: E402  (imported after the dependency checks above)


def test_oracle():
    sanity.check_oracle()


def test_encoders():
    sanity.check_encoders()


def test_dataset():
    sanity.check_dataset()


def test_masked_loss():
    sanity.check_loss()


def test_exprgen_selfcheck():
    sanity.check_exprgen()


def test_tree_lstm_batched_equals_recursive():
    sanity.check_tree_lstm()


def test_stackvm_matches_oracle():
    sanity.check_stackvm()


def test_reducer_matches_oracle():
    sanity.check_reducer()


def test_typed_reducer_matches_oracle_battery():
    sanity.check_typed()


def test_rewrite_soundness_gate():
    sanity.check_rewrite()
