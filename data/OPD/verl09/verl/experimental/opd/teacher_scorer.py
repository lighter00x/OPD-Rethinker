# Copyright 2026 OPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exact teacher log-probability scoring for paper-compatible OPD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl.plugin.platform import get_platform

if TYPE_CHECKING:
    from verl.single_controller.ray.base import RayResourcePool
    from verl.workers.config import DistillationConfig, OPDStudentTopKConfig


def _resolve_dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported OPD teacher dtype: {name!r}") from exc


class _OPDTeacherScorer:
    def __init__(self, model_path: str, config: OPDStudentTopKConfig):
        from transformers import AutoModelForCausalLM

        dtype = _resolve_dtype(config.dtype)
        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": config.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if config.attn_implementation:
            load_kwargs["attn_implementation"] = config.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval().to("cuda")
        self.model.requires_grad_(False)
        self.backbone = getattr(self.model, "model", None)
        self.lm_head = self.model.get_output_embeddings()
        if self.backbone is None or self.lm_head is None:
            raise TypeError("opd_student_topk requires a Hugging Face causal LM with .model and an output embedding.")
        self.temperature = float(config.temperature)
        if self.temperature <= 0:
            raise ValueError(f"OPD teacher temperature must be positive, got {self.temperature}.")
        self.logits_chunk_size = int(config.logits_chunk_size)
        if self.logits_chunk_size <= 0:
            raise ValueError(f"logits_chunk_size must be positive, got {self.logits_chunk_size}.")

    def ready(self) -> bool:
        return True

    @torch.inference_mode()
    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        student_topk_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = torch.device("cuda")
        batch_size, sequence_length = input_ids.shape
        response_length = student_topk_ids.shape[1]
        prompt_width = sequence_length - response_length
        if prompt_width <= 0:
            raise ValueError(
                "Expected a non-empty prompt, "
                f"got sequence_length={sequence_length}, response_length={response_length}."
            )

        outputs = torch.zeros(student_topk_ids.shape, dtype=torch.float32)
        for row in range(batch_size):
            row_input_ids = input_ids[row : row + 1].to(device=device, non_blocking=True)
            row_attention_mask = attention_mask[row : row + 1].to(device=device, non_blocking=True)
            position_ids = (row_attention_mask.long().cumsum(dim=-1) - 1).clamp_min_(0)
            position_ids.masked_fill_(row_attention_mask == 0, 0)

            hidden = self.backbone(
                input_ids=row_input_ids,
                attention_mask=row_attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            response_hidden = hidden[:, prompt_width - 1 : prompt_width + response_length - 1].squeeze(0)
            candidate_ids = student_topk_ids[row].to(device=device, dtype=torch.long, non_blocking=True)
            if candidate_ids.max().item() >= self.lm_head.weight.shape[0] or candidate_ids.min().item() < 0:
                raise ValueError("Student Top-K token IDs are outside the teacher vocabulary.")

            row_output = torch.empty(candidate_ids.shape, dtype=torch.float32, device=device)
            for start in range(0, response_length, self.logits_chunk_size):
                end = min(start + self.logits_chunk_size, response_length)
                logits = self.lm_head(response_hidden[start:end]) / self.temperature
                logits = logits.float()
                selected_logits = logits.gather(dim=-1, index=candidate_ids[start:end])
                row_output[start:end] = selected_logits - torch.logsumexp(logits, dim=-1, keepdim=True)

            valid = response_mask[row].to(device=device, dtype=torch.bool, non_blocking=True).unsqueeze(-1)
            row_output.masked_fill_(~valid, 0.0)
            outputs[row].copy_(row_output.cpu())
            del hidden, response_hidden, row_output

        return outputs


class OPDTeacherScorerManager:
    """Replicate a teacher on a dedicated Ray GPU pool and shard batches over it."""

    def __init__(self, distillation_config: DistillationConfig, resource_pool: RayResourcePool):
        if len(distillation_config.teacher_models) != 1:
            raise ValueError("opd_student_topk currently supports exactly one teacher model.")
        teacher = next(iter(distillation_config.teacher_models.values()))
        scorer_config = distillation_config.opd_student_topk
        if scorer_config.num_scorers != resource_pool.world_size:
            raise ValueError(
                f"Expected {scorer_config.num_scorers} scorer GPUs, got resource pool size {resource_pool.world_size}."
            )

        remote_cls = ray.remote(_OPDTeacherScorer)
        resource_options = get_platform().ray_resource_options(1)
        placement_groups = resource_pool.get_placement_groups(device_name=get_platform().device_name)
        self.workers = []
        for placement_group, bundle_count in zip(placement_groups, resource_pool.store, strict=True):
            for bundle_index in range(bundle_count):
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_index,
                )
                worker = remote_cls.options(
                    num_cpus=1,
                    scheduling_strategy=scheduling_strategy,
                    **resource_options,
                ).remote(teacher.model_path, scorer_config)
                self.workers.append(worker)
        ray.get([worker.ready.remote() for worker in self.workers])

    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        student_topk_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        if batch_size == 0:
            return torch.empty_like(student_topk_ids, dtype=torch.float32)

        num_shards = min(batch_size, len(self.workers))
        boundaries = torch.linspace(0, batch_size, num_shards + 1, dtype=torch.int64).tolist()
        futures = []
        for worker, start, end in zip(self.workers, boundaries[:-1], boundaries[1:], strict=False):
            futures.append(
                worker.score.remote(
                    input_ids[start:end].detach().cpu(),
                    attention_mask[start:end].detach().cpu(),
                    student_topk_ids[start:end].detach().cpu(),
                    response_mask[start:end].detach().cpu(),
                )
            )
        return torch.cat(ray.get(futures), dim=0)
