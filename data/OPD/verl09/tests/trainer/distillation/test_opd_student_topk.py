from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.trainer.distillation.fsdp.losses import compute_opd_student_topk
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def _nested(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    return torch.nested.nested_tensor_from_jagged(values, offsets)


def _reference_opd_loss(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(-1, candidate_ids)
    advantages = (-(old_log_probs - teacher_log_probs) * torch.softmax(old_log_probs, dim=-1)).detach()
    log_ratio = (current_log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(0.8, 1.2)
    losses = torch.maximum(unclipped, clipped)
    losses = torch.where(advantages < 0, torch.minimum(-advantages * 3.0, losses), losses)
    return losses.sum(dim=-1), advantages.sum(dim=-1)


def test_opd_loss_and_gradient_match_released_formula():
    torch.manual_seed(7)
    num_tokens, vocab_size, topk = 7, 19, 4
    offsets = torch.tensor([0, 3, 7], dtype=torch.int64)
    candidate_ids = torch.randint(vocab_size, (1, num_tokens, topk))
    old_logits = torch.randn(1, num_tokens, vocab_size)
    teacher_logits = torch.randn(1, num_tokens, vocab_size)
    old_log_probs = torch.log_softmax(old_logits, dim=-1).gather(-1, candidate_ids)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1).gather(-1, candidate_ids)

    config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            chunked_topk_chunk_size=3,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
        )
    )
    actual_logits = torch.randn(1, num_tokens, vocab_size, requires_grad=True)
    expected_logits = actual_logits.detach().clone().requires_grad_(True)

    actual = compute_opd_student_topk(
        student_logits=actual_logits,
        student_topk_ids=_nested(candidate_ids.squeeze(0), offsets),
        old_student_topk_log_probs=_nested(old_log_probs.squeeze(0), offsets),
        teacher_on_student_log_probs=_nested(teacher_log_probs.squeeze(0), offsets),
        config=config,
        data_format="thd",
    )
    expected_loss, expected_reward = _reference_opd_loss(
        expected_logits, candidate_ids, old_log_probs, teacher_log_probs
    )

    torch.testing.assert_close(actual["distillation_losses"], expected_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual["opd_reward"], expected_reward, atol=1e-6, rtol=1e-6)
    actual["distillation_losses"].sum().backward()
    expected_loss.sum().backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad, atol=1e-6, rtol=1e-6)


def test_response_topk_fields_align_with_prediction_positions():
    input_ids = torch.tensor(
        [
            [0, 11, 12, 13, 21, 22, 0],
            [0, 0, 14, 15, 23, 24, 25],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1],
        ]
    )
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    candidate_ids = torch.tensor(
        [
            [[101, 102], [103, 104], [0, 0]],
            [[201, 202], [203, 204], [205, 206]],
        ]
    )
    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": attention_mask.long().cumsum(-1) - 1,
            "prompts": input_ids[:, :4],
            "responses": input_ids[:, 4:],
            "response_mask": response_mask,
            "opd_student_topk_ids": candidate_ids,
        },
        batch_size=[2],
    )

    no_padding = left_right_2_no_padding(data)
    round_trip = no_padding_2_padding(no_padding["opd_student_topk_ids"], no_padding)

    torch.testing.assert_close(round_trip, candidate_ids)
