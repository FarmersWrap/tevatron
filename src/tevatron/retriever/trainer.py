import os
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel

import logging
logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self._static_graph_set = False

    def _maybe_set_static_graph(self, reason: str) -> None:
        if self._static_graph_set or not dist.is_initialized() or not self.args.gradient_checkpointing:
            return
        candidates = []
        if hasattr(self, "model_wrapped") and self.model_wrapped is not None:
            candidates.append(self.model_wrapped)
        candidates.append(self.model)
        for candidate in candidates:
            if isinstance(candidate, DistributedDataParallel) and hasattr(candidate, "_set_static_graph"):
                candidate._set_static_graph()
                self._static_graph_set = True
                logger.info("Enabled DDP static graph for gradient checkpointing (%s).", reason)
                return

    def _wrap_model(self, model, training=True, dataloader=None):
        model = super()._wrap_model(model, training=training, dataloader=dataloader)
        # dist may be initialized after __init__, so re-check here
        self.is_ddp = dist.is_initialized()
        if training:
            self._maybe_set_static_graph("wrap_model")
        return model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_for_attrs = self.model.module if hasattr(self.model, "module") else self.model
        if not isinstance(model_for_attrs, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if (
                hasattr(model_for_attrs, "query_encoder")
                and model_for_attrs.query_encoder is not model_for_attrs.encoder
            ):
                query_dir = os.path.join(output_dir, "query_encoder")
                passage_dir = os.path.join(output_dir, "passage_encoder")
                os.makedirs(query_dir, exist_ok=True)
                os.makedirs(passage_dir, exist_ok=True)
                if state_dict is None:
                    model_for_attrs.query_encoder.save_pretrained(
                        query_dir, safe_serialization=self.args.save_safetensors
                    )
                    model_for_attrs.encoder.save_pretrained(
                        passage_dir, safe_serialization=self.args.save_safetensors
                    )
                else:
                    query_prefix = "query_encoder."
                    passage_prefix = "encoder."
                    query_state = {
                        k[len(query_prefix):]: v
                        for k, v in state_dict.items()
                        if k.startswith(query_prefix)
                    }
                    passage_state = {
                        k[len(passage_prefix):]: v
                        for k, v in state_dict.items()
                        if k.startswith(passage_prefix)
                    }
                    model_for_attrs.query_encoder.save_pretrained(
                        query_dir, state_dict=query_state, safe_serialization=self.args.save_safetensors
                    )
                    model_for_attrs.encoder.save_pretrained(
                        passage_dir, state_dict=passage_state, safe_serialization=self.args.save_safetensors
                    )
            else:
                if state_dict is None:
                    state_dict = model_for_attrs.state_dict()
                prefix = 'encoder.'
                assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
                state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
                model_for_attrs.encoder.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage, *rest = inputs
        eos_positions = rest[0] if rest else None
        # input(f"trainer.compute_loss: eos_positions: {eos_positions}")
        model_for_attrs = model.module if hasattr(model, "module") else model
        if hasattr(model_for_attrs, "eos_positions"):
            model_for_attrs.eos_positions = eos_positions
        if getattr(model_for_attrs, "passage_chunk_size", 0) > 0 and eos_positions is None:
            logger.warning(
                "Chunked training enabled (passage_chunk_size=%s) but eos_positions is None. "
                "MaxSim will be disabled for this batch.",
                getattr(model_for_attrs, "passage_chunk_size", 0),
            )
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        self._maybe_set_static_graph("training_step")
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor


class DistilTevatronTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage, reranker_labels = inputs
        scores = model(query=query, passage=passage).scores
        
        if model.is_ddp:
            # reranker_scores are gathered across all processes
            reranker_labels = model._dist_gather_tensor(reranker_labels)
        
        # Derive student_scores [batch, num_labels]
        batch_size, total_passages = scores.size()
        num_labels = reranker_labels.size(1)
        start_idxs = torch.arange(0, batch_size * num_labels, num_labels, device=scores.device)
        idx_matrix = start_idxs.view(-1, 1) + torch.arange(num_labels, device=scores.device)
        student_scores = scores.gather(1, idx_matrix)

        # Temperature‐scaled soft distributions
        T = self.args.distil_temperature
        student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
        teacher_probs = torch.softmax(reranker_labels.float()    / T, dim=1)

        # KL Divergence loss (shapes now [batch, num_labels])
        loss = torch.nn.functional.kl_div(
            student_log,
            teacher_probs,
            reduction="batchmean"
        ) * self._dist_loss_scale_factor

        return loss

    def training_step(self, *args):
        return super(DistilTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor
