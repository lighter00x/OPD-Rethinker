# Copyright 2026

"""Dynamic metrics for paper-compatible student Top-K OPD."""

from __future__ import annotations

import torch


def compute_opd_dynamic_metrics(
    student_topk_ids: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    student_entropy: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_size: int = 8192,
) -> dict[str, float]:
    """Compute Section 2.3 OPD metrics on valid response positions."""
    if student_topk_ids.shape != student_topk_log_probs.shape:
        raise ValueError("student top-k IDs and log-probabilities must have the same shape")
    if teacher_topk_ids.shape != teacher_topk_log_probs.shape:
        raise ValueError("teacher top-k IDs and log-probabilities must have the same shape")
    if student_topk_ids.shape != teacher_topk_ids.shape:
        raise ValueError("student and teacher top-k tensors must have the same shape")
    if student_topk_ids.ndim != 3:
        raise ValueError(f"expected [batch, response, k] top-k tensors, got {student_topk_ids.shape}")
    if student_entropy.shape != response_mask.shape or teacher_entropy.shape != response_mask.shape:
        raise ValueError("entropy tensors and response mask must have shape [batch, response]")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    valid = response_mask.bool().reshape(-1)
    if not valid.any():
        return {}

    k = student_topk_ids.shape[-1]
    student_ids = student_topk_ids.reshape(-1, k)[valid].long()
    teacher_ids = teacher_topk_ids.reshape(-1, k)[valid].long()
    student_logp = student_topk_log_probs.reshape(-1, k)[valid].float()
    teacher_logp = teacher_topk_log_probs.reshape(-1, k)[valid].float()
    student_h = student_entropy.reshape(-1)[valid].float()
    teacher_h = teacher_entropy.reshape(-1)[valid].float()

    overlap_count_sum = 0.0
    overlap_advantage_sum = 0.0
    overlap_positions = 0
    extreme_advantage_min = float("inf")
    extreme_advantage_max = float("-inf")
    extreme_min_prob_diff = 0.0
    extreme_max_prob_diff = 0.0

    for start in range(0, student_ids.shape[0], chunk_size):
        end = min(start + chunk_size, student_ids.shape[0])
        sid = student_ids[start:end]
        tid = teacher_ids[start:end]
        slogp = student_logp[start:end]
        tlogp = teacher_logp[start:end]

        match = tid.unsqueeze(-1).eq(sid.unsqueeze(-2))
        teacher_overlap = match.any(dim=-1)
        overlap_count = teacher_overlap.sum(dim=-1)
        overlap_count_sum += overlap_count.sum().item()

        student_logp_bar = slogp - torch.logsumexp(slogp, dim=-1, keepdim=True)
        teacher_logp_bar = tlogp - torch.logsumexp(tlogp, dim=-1, keepdim=True)
        matched_student_logp = torch.where(
            match, student_logp_bar.unsqueeze(-2), torch.zeros_like(match, dtype=student_logp_bar.dtype)
        ).sum(dim=-1)
        matched_student_prob = matched_student_logp.exp()
        teacher_prob = teacher_logp_bar.exp()
        token_advantage = matched_student_prob * (teacher_logp_bar - matched_student_logp)
        token_advantage = token_advantage.masked_fill(~teacher_overlap, 0.0)

        has_overlap = overlap_count > 0
        if has_overlap.any():
            per_position_advantage = token_advantage.sum(dim=-1) / overlap_count.clamp_min(1)
            overlap_advantage_sum += per_position_advantage[has_overlap].sum().item()
            overlap_positions += has_overlap.sum().item()

            valid_advantage = token_advantage[teacher_overlap]
            valid_prob_diff = (teacher_prob - matched_student_prob)[teacher_overlap]
            local_min, local_min_idx = valid_advantage.min(dim=0)
            local_max, local_max_idx = valid_advantage.max(dim=0)
            if local_min.item() < extreme_advantage_min:
                extreme_advantage_min = local_min.item()
                extreme_min_prob_diff = valid_prob_diff[local_min_idx].item()
            if local_max.item() > extreme_advantage_max:
                extreme_advantage_max = local_max.item()
                extreme_max_prob_diff = valid_prob_diff[local_max_idx].item()

    num_positions = student_ids.shape[0]
    metrics = {
        "distillation/overlap_ratio": overlap_count_sum / (num_positions * k),
        "distillation/overlap_count": overlap_count_sum / num_positions,
        "distillation/overlap_token_advantage": (
            overlap_advantage_sum / overlap_positions if overlap_positions else 0.0
        ),
        "distillation/overlap_zero_fraction": 1.0 - (overlap_positions / num_positions),
        "distillation/student_entropy": student_h.mean().item(),
        "distillation/teacher_entropy": teacher_h.mean().item(),
        "distillation/entropy_gap": (teacher_h - student_h).abs().mean().item(),
    }
    if overlap_positions:
        metrics.update(
            {
                "distillation/extreme_advantage_min": extreme_advantage_min,
                "distillation/extreme_advantage_max": extreme_advantage_max,
                "distillation/extreme_advantage_min_prob_diff": extreme_min_prob_diff,
                "distillation/extreme_advantage_max_prob_diff": extreme_max_prob_diff,
            }
        )
    return metrics
