import torch
import torch.nn.functional as F
from transformers import Trainer


def _safe_to_device(tensor_or_model, device):
    """Move a tensor or model to device, falling back to CPU if MPS fails."""
    try:
        return tensor_or_model.to(device)
    except RuntimeError:
        return tensor_or_model.to("cpu")


class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, beta=0.3, temperature=2.0,
                 use_sequence_distill=False, seq_kd=0.1,
                 use_topk_distill=False, topk_kd_weight=0.2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.seq_kd = seq_kd
        self.use_sequence_distill = use_sequence_distill
        self.use_topk_distill = use_topk_distill
        self.topk_kd_weight = topk_kd_weight

        total = alpha + beta + (seq_kd if use_sequence_distill else 0.0)
        if total > 1.0:
            raise ValueError("alpha + beta (+ seq_kd) must be <= 1.0")
        self.kd_weight = 1.0 - alpha - beta - (seq_kd if use_sequence_distill else 0.0)

        if self.teacher_model is not None:
            self.teacher_model.eval()

    # ── device helpers ──────────────────────────────────────────────
    @staticmethod
    def _current_device(model):
        return next(model.parameters()).device

    def _move_teacher_to_device(self, device):
        if self.teacher_model is None:
            return
        if self._current_device(self.teacher_model) != device:
            self.teacher_model = _safe_to_device(self.teacher_model, device)

    @staticmethod
    def _is_mps(device):
        return hasattr(device, "type") and device.type == "mps"

    # ── MPS-safe KL divergence (full vocab, live teacher) ───────────
    @staticmethod
    def _kl_divergence(student_log_probs, teacher_probs, mask, temperature):
        teacher_log_probs = torch.clamp(teacher_probs.log(), min=-1e4)
        per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        per_token = per_token * mask
        kd_loss = per_token.sum() / mask.sum().clamp(min=1)
        return kd_loss * (temperature ** 2)

    # ── Sparse KL from offline top-k logprobs ───────────────────────
    @staticmethod
    def _topk_kl_divergence(student_logits, topk_token_ids, topk_logprobs, mask, temperature):
        """
        KL divergence using only the top-k teacher probabilities.
        topk_token_ids: (batch, seq_len, k) — token indices
        topk_logprobs:  (batch, seq_len, k) — teacher log probabilities
        """
        # Teacher: renormalize top-k into a distribution
        teacher_topk_probs = F.softmax(topk_logprobs / temperature, dim=-1)

        # Student: gather log-probs at same token positions
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
        student_topk_lp = torch.gather(student_log_probs, dim=-1, index=topk_token_ids)

        # KL over top-k subset
        per_token = (teacher_topk_probs * (teacher_topk_probs.log() - student_topk_lp)).sum(dim=-1)
        per_token = per_token * mask
        kl = per_token.sum() / mask.sum().clamp(min=1)
        return kl * (temperature ** 2)

    # ── sequence distillation (live teacher generate → student CE) ──
    def _sequence_distill_loss(self, model, inputs, labels, device):
        input_len = inputs["input_ids"].shape[1]
        pad_id = self.teacher_model.config.pad_token_id or self.teacher_model.config.eos_token_id

        gen_device = torch.device("cpu") if self._is_mps(device) else device

        with torch.no_grad():
            gen_ids = inputs["input_ids"].to(gen_device)
            gen_attn = inputs.get("attention_mask")
            if gen_attn is not None:
                gen_attn = gen_attn.to(gen_device)
            teacher = _safe_to_device(self.teacher_model, gen_device)
            teacher_gen = teacher.generate(
                input_ids=gen_ids, attention_mask=gen_attn,
                max_new_tokens=labels.shape[1], do_sample=False, pad_token_id=pad_id,
            ).to(device)

        if gen_device != device:
            self.teacher_model = _safe_to_device(self.teacher_model, device)

        seq_labels = teacher_gen.clone()
        seq_labels[:, :input_len] = -100
        seq_labels[teacher_gen == pad_id] = -100
        seq_attn = (teacher_gen != pad_id).long()

        out = model(input_ids=teacher_gen, attention_mask=seq_attn)
        logits = out.logits[:, :-1, :].contiguous()
        seq_labels = seq_labels[:, 1:].contiguous()

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), seq_labels.view(-1), ignore_index=-100,
        )

    # ── main loss ───────────────────────────────────────────────────
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = self._current_device(model)
        self._move_teacher_to_device(device)

        labels = inputs.pop("labels")
        rationale_mask = inputs.pop("rationale_mask", None)
        topk_token_ids = inputs.pop("topk_token_ids", None)
        topk_logprobs = inputs.pop("topk_logprobs", None)

        # ---- student forward ----
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        if rationale_mask is not None:
            rationale_mask = rationale_mask[:, 1:].contiguous()

        # ---- answer / rationale CE losses ----
        if rationale_mask is not None:
            ans_labels = labels.clone()
            ans_labels[rationale_mask.bool()] = -100
            answer_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                ans_labels.view(-1), ignore_index=-100,
            )
            rat_labels = labels.clone()
            rat_labels[~rationale_mask.bool()] = -100
            rationale_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                rat_labels.view(-1), ignore_index=-100,
            )
        else:
            answer_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1), ignore_index=-100,
            )
            rationale_loss = torch.tensor(0.0, device=device)

        total_loss = self.alpha * answer_loss + self.beta * rationale_loss

        # ---- offline top-k logit distillation (from API) ----
        if self.use_topk_distill and topk_token_ids is not None:
            topk_token_ids = topk_token_ids[:, 1:, :].contiguous()
            topk_logprobs = topk_logprobs[:, 1:, :].contiguous()
            mask = (labels != -100).float()
            topk_loss = self._topk_kl_divergence(
                student_logits, topk_token_ids, topk_logprobs, mask, self.temperature,
            )
            total_loss = total_loss + self.topk_kd_weight * topk_loss

        # ---- live teacher logit distillation ----
        if self.teacher_model is not None:
            with torch.no_grad():
                t_out = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                t_logits = t_out.logits[:, :-1, :].contiguous()

            mask = (labels != -100).float()
            s_lp = F.log_softmax(student_logits.float() / self.temperature, dim=-1)
            t_p = F.softmax(t_logits.float() / self.temperature, dim=-1)
            kd_loss = self._kl_divergence(s_lp, t_p, mask, self.temperature)
            total_loss = total_loss + self.kd_weight * kd_loss

        # ---- live sequence distillation ----
        if self.use_sequence_distill and self.teacher_model is not None:
            total_loss = total_loss + self.seq_kd * self._sequence_distill_loss(
                model, inputs, labels, device,
            )

        return (total_loss, student_outputs) if return_outputs else total_loss
