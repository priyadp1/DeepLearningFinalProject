import torch
import torch.nn.functional as F
from transformers import Trainer

# Custom KDTrainer that incorporates answer loss, rationale loss, and logit alignment loss
class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, beta=0.3, temperature=2.0,
                 use_sequence_distill=False, seq_kd=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha # weight for answer loss
        self.beta = beta # weight for rationale loss
        self.temperature = temperature
        self.seq_kd = seq_kd # weight for sequence distillation loss
        self.use_sequence_distill = use_sequence_distill

        total = alpha + beta + (seq_kd if use_sequence_distill else 0.0)
        if total > 1.0:
            raise ValueError("alpha + beta (+ seq_kd when use_sequence_distill=True) must be <= 1.0")
        self.kd_weight = 1.0 - alpha - beta - (seq_kd if use_sequence_distill else 0.0)

        if self.teacher_model is not None:
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        # Remove rationale_mask from inputs if it exists
        rationale_mask = inputs.pop("rationale_mask", None)

        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        student_logits = student_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        if rationale_mask is not None:
            rationale_mask = rationale_mask[:, 1:].contiguous()

        # Compute answer and rationale losses
        if rationale_mask is not None:
            answer_labels = labels.clone()
            answer_labels[rationale_mask.bool()] = -100
            answer_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                answer_labels.view(-1),
                ignore_index=-100,
            )

            # Rationale loss only on rationale tokens
            rationale_labels = labels.clone()
            rationale_labels[~rationale_mask.bool()] = -100
            rationale_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                rationale_labels.view(-1),
                ignore_index=-100,
            )
        else:
            # If no rationale_mask, treat all tokens as answer tokens
            answer_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            rationale_loss = torch.tensor(0.0, device=answer_loss.device)

        if self.teacher_model is None:
            total_loss = self.alpha * answer_loss + self.beta * rationale_loss
            return (total_loss, student_outputs) if return_outputs else total_loss

        # Single teacher forward pass (reused for logit distillation)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )
            teacher_logits = teacher_outputs.logits

        # Logit distillation: token-level KL divergence
        t_logits = teacher_logits[:, :-1, :].contiguous()
        mask = labels != -100
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(t_logits / self.temperature, dim=-1)
        kd_loss_fct = torch.nn.KLDivLoss(reduction="none")
        kd_loss = kd_loss_fct(student_log_probs, teacher_probs).sum(dim=-1)
        kd_loss = (kd_loss * mask).sum() / mask.sum().clamp(min=1)
        kd_loss = kd_loss * (self.temperature ** 2)

        total_loss = (
            self.alpha * answer_loss
            + self.beta * rationale_loss
            + self.kd_weight * kd_loss
        )

        # Sequence distillation
        if self.use_sequence_distill:
            input_len = inputs["input_ids"].shape[1]
            pad_token_id = (
                self.teacher_model.config.pad_token_id
                or self.teacher_model.config.eos_token_id
            )
            with torch.no_grad():
                teacher_gen_ids = self.teacher_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=labels.shape[1],
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )

            # Mask input tokens; only supervise on generated tokens
            seq_labels = teacher_gen_ids.clone()
            seq_labels[:, :input_len] = -100
            seq_labels[teacher_gen_ids == pad_token_id] = -100

            seq_attn_mask = (teacher_gen_ids != pad_token_id).long()

            seq_student_outputs = model(
                input_ids=teacher_gen_ids,
                attention_mask=seq_attn_mask,
            )
            seq_student_logits = seq_student_outputs.logits[:, :-1, :].contiguous()
            seq_labels_shifted = seq_labels[:, 1:].contiguous()

            seq_kd_loss = F.cross_entropy(
                seq_student_logits.view(-1, seq_student_logits.size(-1)),
                seq_labels_shifted.view(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + self.seq_kd * seq_kd_loss

        # Return the total loss and student outputs if requested
        return (total_loss, student_outputs) if return_outputs else total_loss
