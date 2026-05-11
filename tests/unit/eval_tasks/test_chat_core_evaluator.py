"""
Unit tests for ChatCoreEvaluator.generate_completion and generate_completion_with_tools.

Test strategy:
- All kv_cache_utils functions and use_calculator are mocked to isolate the state
  machine logic inside the two generation methods.
- Behavioral contracts are verified: which tokens land in the final decode call,
  when generation terminates, and how the tool-use state machine handles python
  blocks, calculator execution, and forced token injection.
- No shallow "it ran without error" tests — every assertion checks a distinct
  behavioral guarantee.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch

src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# wandb is required by eval_tasks.core but not available in unit test environments
if "wandb" not in sys.modules:
    sys.modules["wandb"] = MagicMock()

from eval_tasks.chat_core.evaluator import ChatCoreEvaluator

MODULE = "eval_tasks.chat_core.evaluator"

# Stable special token IDs used throughout
ASSISTANT_END = 1004
PYTHON_START = 1000
PYTHON_END = 1001
OUTPUT_START = 1002
OUTPUT_END = 1003

SPECIAL_TOKENS = {
    "<|python|>": PYTHON_START,
    "<|python_end|>": PYTHON_END,
    "<|output_start|>": OUTPUT_START,
    "<|output_end|>": OUTPUT_END,
    "<|assistant_end|>": ASSISTANT_END,
}

# A logits tensor — content doesn't matter since sample_next_token is mocked
DUMMY_LOGITS = torch.zeros(512)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokenizer(with_tools: bool = True) -> MagicMock:
    tok = MagicMock()
    tok._special_tokens = SPECIAL_TOKENS.copy() if with_tools else {}
    tok.decode.return_value = "decoded text"
    tok.encode.return_value = [99]
    return tok


def _make_evaluator(
    with_tools: bool = True,
    max_tokens: int = 30,
    use_kv_cache: bool = True,
) -> ChatCoreEvaluator:
    model = MagicMock()
    model.max_seq_len = 1024
    return ChatCoreEvaluator(
        model=model,
        tokenizer=_make_tokenizer(with_tools),
        device=torch.device("cpu"),
        master_process=False,
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=50,
        use_kv_cache=use_kv_cache,
    )


# ---------------------------------------------------------------------------
# generate_completion
# ---------------------------------------------------------------------------


class TestGenerateCompletion:
    """Tests for the basic (no-tool) generation loop."""

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    def test_stops_at_assistant_end_token(
        self, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """Generation stops when assistant_end is sampled; end token excluded from output."""
        mock_sample.side_effect = [42, 43, ASSISTANT_END]

        ev = _make_evaluator()
        ev.generate_completion([10, 20])

        decoded_tokens = ev.tokenizer.decode.call_args[0][0]
        assert decoded_tokens == [42, 43], "End token must not appear in decoded output"
        # forward_pass should be called only for token 42 and 43, not after end token
        assert mock_forward.call_count == 2

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token", return_value=7)
    def test_stops_at_max_tokens(
        self, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """Generation halts exactly at max_tokens when no end token is ever sampled."""
        max_tokens = 5
        ev = _make_evaluator(max_tokens=max_tokens)
        ev.generate_completion([1])

        decoded_tokens = ev.tokenizer.decode.call_args[0][0]
        assert len(decoded_tokens) == max_tokens

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    def test_excludes_prompt_from_decoded_output(
        self, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """Decode receives only the newly generated tokens, never the prompt tokens."""
        mock_sample.side_effect = [200, 201, ASSISTANT_END]
        prompt = [10, 20, 30]

        ev = _make_evaluator()
        ev.generate_completion(prompt)

        decoded_tokens = ev.tokenizer.decode.call_args[0][0]
        for pt in prompt:
            assert pt not in decoded_tokens, f"Prompt token {pt} leaked into decode call"
        assert decoded_tokens == [200, 201]

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    def test_forward_pass_receives_single_int_with_kv_cache(
        self, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """With KV cache enabled, each decode step passes a single token (int) to forward_pass."""
        mock_sample.side_effect = [77, 88, ASSISTANT_END]

        ev = _make_evaluator(use_kv_cache=True)
        ev.generate_completion([1, 2])

        assert mock_forward.call_count == 2
        # Second positional arg to each call must be an integer (single token)
        for c in mock_forward.call_args_list:
            assert isinstance(c[0][1], int), "KV-cache mode must pass a single int token"

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache", return_value=None)
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    def test_forward_pass_receives_full_sequence_without_kv_cache(
        self, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """Without KV cache, forward_pass receives the full accumulated token list."""
        mock_sample.side_effect = [77, ASSISTANT_END]

        ev = _make_evaluator(use_kv_cache=False)
        ev.generate_completion([1, 2])

        token_arg = mock_forward.call_args[0][1]
        assert isinstance(token_arg, list), "No-cache mode must pass the full token list"
        assert 77 in token_arg
        assert 1 in token_arg and 2 in token_arg

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token", return_value=ASSISTANT_END)
    def test_empty_output_when_first_token_is_end(
        self, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """When the very first sampled token is the end token, no forward_pass calls occur."""
        ev = _make_evaluator()
        ev.generate_completion([5, 6, 7])

        decoded_tokens = ev.tokenizer.decode.call_args[0][0]
        assert decoded_tokens == []
        mock_forward.assert_not_called()

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 10))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token", return_value=5)
    def test_stops_at_model_max_seq_len(
        self, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """Generation stops when total sequence length hits model.max_seq_len."""
        ev = _make_evaluator(max_tokens=100)
        ev.model.max_seq_len = 10
        prompt = [1, 2, 3, 4, 5]  # 5 tokens; model limit is 10

        ev.generate_completion(prompt)

        decoded_tokens = ev.tokenizer.decode.call_args[0][0]
        # After appending a new token the check fires when len(generated_tokens) >= 10,
        # so at most 5 new tokens can be produced (5 prompt + 5 new = 10).
        assert len(decoded_tokens) <= 5


# ---------------------------------------------------------------------------
# generate_completion_with_tools
# ---------------------------------------------------------------------------


class TestGenerateCompletionWithTools:
    """Tests for the tool-use generation loop (python block state machine)."""

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    def test_fallback_to_regular_generation_when_no_tool_support(
        self, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """When the tokenizer lacks tool special tokens, falls back to generate_completion.

        With no special tokens, _get_assistant_end_token() returns the GPT-2 fallback
        (50256), so we must use that as the termination signal in the mock.
        """
        GPT2_EOT = 50256  # fallback when _special_tokens is missing <|assistant_end|>
        mock_sample.side_effect = [42, GPT2_EOT]

        # Empty special_tokens dict → _check_tool_support returns False → supports_tools=False
        ev = _make_evaluator(with_tools=False)
        assert not ev.supports_tools

        ev.generate_completion_with_tools([1, 2])

        # Must still produce output via the regular path
        decoded_tokens = ev.tokenizer.decode.call_args[0][0]
        assert decoded_tokens == [42]

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    @patch(f"{MODULE}.use_calculator")
    def test_plain_generation_without_python_block(
        self, mock_calc, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """Normal tokens pass through unchanged; calculator is never invoked."""
        mock_sample.side_effect = [10, 20, 30, ASSISTANT_END]

        ev = _make_evaluator()
        ev.generate_completion_with_tools([1])

        mock_calc.assert_not_called()
        decoded_tokens = ev.tokenizer.decode.call_args[0][0]
        assert decoded_tokens == [10, 20, 30]

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    @patch(f"{MODULE}.use_calculator")
    def test_tool_use_injects_calculator_result(
        self, mock_calc, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """A python block triggers calculator execution and injects output tokens into the sequence."""
        expr_token = 50  # token inside the python block
        mock_sample.side_effect = [
            PYTHON_START,
            expr_token,
            PYTHON_END,
            ASSISTANT_END,
        ]
        mock_calc.return_value = 42

        ev = _make_evaluator()
        # tokenizer.decode: first call decodes expr_tokens, second decodes final output
        ev.tokenizer.decode.side_effect = ["12+30", "final"]
        # tokenizer.encode: encode the result string "42"
        ev.tokenizer.encode.return_value = [55, 56]

        ev.generate_completion_with_tools([1])

        # Calculator called with the decoded expression
        mock_calc.assert_called_once_with("12+30")

        # Final decoded sequence must include the forced output tokens
        decoded_tokens = ev.tokenizer.decode.call_args_list[-1][0][0]
        assert OUTPUT_START in decoded_tokens
        assert OUTPUT_END in decoded_tokens
        # Encoded result tokens injected between markers
        assert 55 in decoded_tokens and 56 in decoded_tokens

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    @patch(f"{MODULE}.use_calculator")
    def test_tool_use_skips_output_when_calculator_returns_none(
        self, mock_calc, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """When use_calculator returns None, no output tokens are forced into the sequence."""
        expr_token = 50
        mock_sample.side_effect = [
            PYTHON_START,
            expr_token,
            PYTHON_END,
            ASSISTANT_END,
        ]
        mock_calc.return_value = None

        ev = _make_evaluator()
        ev.tokenizer.decode.side_effect = ["bad_expr", "final"]

        ev.generate_completion_with_tools([1])

        decoded_tokens = ev.tokenizer.decode.call_args_list[-1][0][0]
        assert OUTPUT_START not in decoded_tokens
        assert OUTPUT_END not in decoded_tokens
        ev.tokenizer.encode.assert_not_called()

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    @patch(f"{MODULE}.use_calculator")
    def test_multiple_sequential_tool_uses(
        self, mock_calc, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """Two consecutive python blocks are each executed and their results injected."""
        mock_sample.side_effect = [
            PYTHON_START, 50, PYTHON_END,   # first tool call
            PYTHON_START, 51, PYTHON_END,   # second tool call
            ASSISTANT_END,
        ]
        mock_calc.side_effect = [10, 20]

        ev = _make_evaluator(max_tokens=50)
        ev.tokenizer.decode.side_effect = ["expr1", "expr2", "final output"]
        ev.tokenizer.encode.side_effect = [[11], [22]]

        ev.generate_completion_with_tools([1])

        assert mock_calc.call_count == 2
        mock_calc.assert_any_call("expr1")
        mock_calc.assert_any_call("expr2")

        decoded_tokens = ev.tokenizer.decode.call_args_list[-1][0][0]
        # Both output blocks must be present
        assert decoded_tokens.count(OUTPUT_START) == 2
        assert decoded_tokens.count(OUTPUT_END) == 2

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    @patch(f"{MODULE}.use_calculator")
    def test_forced_tokens_fed_through_kv_cache(
        self, mock_calc, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """Each forced result token triggers a forward_pass so the KV cache stays up to date."""
        expr_token = 50
        mock_sample.side_effect = [PYTHON_START, expr_token, PYTHON_END, ASSISTANT_END]
        mock_calc.return_value = 5
        result_tokens = [77, 78, 79]

        ev = _make_evaluator()
        ev.tokenizer.decode.side_effect = ["expr", "final"]
        ev.tokenizer.encode.return_value = result_tokens

        # Count forward_pass calls before and after
        ev.generate_completion_with_tools([1])

        forward_call_count = mock_forward.call_count
        # 3 normal sampled tokens (PYTHON_START, expr_token, PYTHON_END) each call forward_pass,
        # plus 3 forced tokens (OUTPUT_START, result*3, OUTPUT_END) = OUTPUT_START + 3 result + OUTPUT_END = 5
        # Total forced = 1 (output_start) + 3 (result) + 1 (output_end) = 5
        # Total from sampled = PYTHON_START (1) + expr_token (1) + PYTHON_END (1) = 3
        # Grand total = 8 forward_pass calls (before ASSISTANT_END terminates)
        forced_count = 1 + len(result_tokens) + 1  # OUTPUT_START + result_tokens + OUTPUT_END
        assert forward_call_count == 3 + forced_count

    @patch(f"{MODULE}.get_model_config", return_value=(4, 32, 2, 512))
    @patch(f"{MODULE}.create_kv_cache")
    @patch(f"{MODULE}.prefill_prompt", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.forward_pass", return_value=DUMMY_LOGITS)
    @patch(f"{MODULE}.sample_next_token")
    @patch(f"{MODULE}.use_calculator")
    def test_python_block_with_empty_expression_skips_calculator(
        self, mock_calc, mock_sample, mock_forward, mock_prefill, mock_create, mock_cfg
    ):
        """An immediately-closed python block (no expr tokens) does not call the calculator."""
        mock_sample.side_effect = [PYTHON_START, PYTHON_END, ASSISTANT_END]

        ev = _make_evaluator()
        ev.tokenizer.decode.return_value = "final"

        ev.generate_completion_with_tools([1])

        mock_calc.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
