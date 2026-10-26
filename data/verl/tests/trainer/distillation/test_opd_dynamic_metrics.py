import pytest
import torch

from verl.experimental.opd.metrics import compute_opd_dynamic_metrics


def test_opd_dynamic_metrics_match_paper_definitions():
    student_ids = torch.tensor([[[1, 2], [3, 4], [0, 0]]])
    teacher_ids = torch.tensor([[[2, 1], [3, 5], [0, 0]]])
    student_logp = torch.log(torch.tensor([[[0.6, 0.3], [0.5, 0.2], [1.0, 1.0]]]))
    teacher_logp = torch.log(torch.tensor([[[0.4, 0.5], [0.6, 0.1], [1.0, 1.0]]]))
    student_entropy = torch.tensor([[1.0, 2.0, 0.0]])
    teacher_entropy = torch.tensor([[1.5, 1.0, 0.0]])
    response_mask = torch.tensor([[1, 1, 0]])

    metrics = compute_opd_dynamic_metrics(
        student_ids,
        student_logp,
        student_entropy,
        teacher_ids,
        teacher_logp,
        teacher_entropy,
        response_mask,
        chunk_size=1,
    )

    # Position overlaps are 2/2 and 1/2, hence E[|intersection|/k] = 0.75.
    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.75)
    assert metrics["distillation/overlap_count"] == pytest.approx(1.5)
    assert metrics["distillation/overlap_zero_fraction"] == pytest.approx(0.0)
    assert metrics["distillation/student_entropy"] == pytest.approx(1.5)
    assert metrics["distillation/teacher_entropy"] == pytest.approx(1.25)
    assert metrics["distillation/entropy_gap"] == pytest.approx(0.75)

    student_bar = torch.softmax(student_logp[:, :2], dim=-1)
    teacher_bar = torch.softmax(teacher_logp[:, :2], dim=-1)
    pos0 = (
        student_bar[0, 0, 1] * (teacher_bar[0, 0, 0].log() - student_bar[0, 0, 1].log())
        + student_bar[0, 0, 0] * (teacher_bar[0, 0, 1].log() - student_bar[0, 0, 0].log())
    ) / 2
    pos1 = student_bar[0, 1, 0] * (teacher_bar[0, 1, 0].log() - student_bar[0, 1, 0].log())
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(((pos0 + pos1) / 2).item())


def test_opd_dynamic_metrics_ignore_padding_and_handle_zero_overlap():
    metrics = compute_opd_dynamic_metrics(
        student_topk_ids=torch.tensor([[[1, 2], [0, 0]]]),
        student_topk_log_probs=torch.tensor([[[-0.1, -2.0], [0.0, 0.0]]]),
        student_entropy=torch.tensor([[0.7, 99.0]]),
        teacher_topk_ids=torch.tensor([[[3, 4], [0, 0]]]),
        teacher_topk_log_probs=torch.tensor([[[-0.2, -1.0], [0.0, 0.0]]]),
        teacher_entropy=torch.tensor([[1.2, 99.0]]),
        response_mask=torch.tensor([[1, 0]]),
    )

    assert metrics["distillation/overlap_ratio"] == 0.0
    assert metrics["distillation/overlap_token_advantage"] == 0.0
    assert metrics["distillation/overlap_zero_fraction"] == 1.0
    assert metrics["distillation/entropy_gap"] == pytest.approx(0.5)
    assert "distillation/extreme_advantage_min" not in metrics
