"""Unit tests for vLLM-backed runners (no server required – uses mocks)."""

from unittest.mock import patch, MagicMock
import pytest

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.vllm.psyllm_gml import PsyLLMVLLMRunner
from reliable_clinical_benchmark.models.vllm.piaget_8b import Piaget8BVLLMRunner
from reliable_clinical_benchmark.models.vllm.psyche_r1 import PsycheR1VLLMRunner
from reliable_clinical_benchmark.models.vllm.psych_qwen_32b import PsychQwen32BVLLMRunner


# ── Factory wiring ──────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id,expected_cls",
    [
        ("psyllm_gml_vllm", PsyLLMVLLMRunner),
        ("psyllm-vllm", PsyLLMVLLMRunner),
        ("piaget_vllm", Piaget8BVLLMRunner),
        ("piaget-8b-vllm", Piaget8BVLLMRunner),
        ("psyche_r1_vllm", PsycheR1VLLMRunner),
        ("psyche-r1-vllm", PsycheR1VLLMRunner),
        ("psych_qwen_vllm", PsychQwen32BVLLMRunner),
        ("psych-qwen-32b-vllm", PsychQwen32BVLLMRunner),
    ],
)
def test_factory_returns_correct_vllm_runner(model_id, expected_cls):
    runner = get_model_runner(model_id)
    assert isinstance(runner, expected_cls)


# ── Default attributes ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_psyllm_vllm_defaults():
    runner = PsyLLMVLLMRunner()
    assert runner.model_name == "GMLHUHE/PsyLLM-8B"
    assert runner.api_base == "http://127.0.0.1:8101/v1"


@pytest.mark.unit
def test_piaget_vllm_defaults():
    runner = Piaget8BVLLMRunner()
    assert runner.model_name == "gustavecortal/Piaget-8B"
    assert runner.api_base == "http://127.0.0.1:8102/v1"


@pytest.mark.unit
def test_psyche_r1_vllm_defaults():
    runner = PsycheR1VLLMRunner()
    assert runner.model_name == "MindIntLab/Psyche-R1"
    assert runner.api_base == "http://127.0.0.1:8103/v1"


@pytest.mark.unit
def test_psych_qwen_vllm_defaults():
    runner = PsychQwen32BVLLMRunner()
    assert runner.model_name == "Compumacy/Psych_Qwen_32B"
    assert runner.api_base == "http://127.0.0.1:8104/v1"


# ── generate() delegates to chat_completion ──────────────────────────────────────


@pytest.mark.unit
@patch("reliable_clinical_benchmark.models.vllm.psyllm_gml.chat_completion")
def test_psyllm_vllm_generate_calls_chat_completion(mock_cc):
    mock_cc.return_value = "Mock response"
    runner = PsyLLMVLLMRunner()
    result = runner.generate("test prompt")
    assert result == "Mock response"
    mock_cc.assert_called_once()
    call_kwargs = mock_cc.call_args
    assert call_kwargs.kwargs["api_base"] == "http://127.0.0.1:8101/v1"
    assert call_kwargs.kwargs["model"] == "GMLHUHE/PsyLLM-8B"
    assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "test prompt"}]


@pytest.mark.unit
@patch("reliable_clinical_benchmark.models.vllm.piaget_8b.chat_completion")
def test_piaget_vllm_generate_calls_chat_completion(mock_cc):
    mock_cc.return_value = "Piaget says hello"
    runner = Piaget8BVLLMRunner()
    result = runner.generate("hello")
    assert result == "Piaget says hello"
    mock_cc.assert_called_once()
