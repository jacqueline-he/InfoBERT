# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import json
import logging
import math
import os
import random
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from packaging import version
from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

from typing import Callable, Dict, List, Optional, Tuple

from transformers import AdamW, get_linear_schedule_with_warmup

from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput, speed_metrics
from transformers.trainer_pt_utils import distributed_concat
from tqdm.auto import tqdm, trange

import torch


from advtraining_args import TrainingArguments, is_tpu_available

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

logger = logging.getLogger(__name__)


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, mi_estimator = None, mi_upper_estimator=None, optimizers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.mi_estimator = mi_estimator
        self.mi_upper_estimator = mi_upper_estimator
        self.eval_hist = []
        self.optimizers = optimizers
        self.prediction_loss_only = False

        # Create output directory if needed
        # is_world_process_zero is inherited from Trainer class
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)

    # INFOBERT STUFF # 
    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        if self.mi_estimator:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)] +
                    list(self.mi_estimator.parameters()),
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def get_mi_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = [
            {
                "params": list(self.mi_estimator.parameters()),
                "weight_decay": self.args.weight_decay,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler


    def train_mi_only(self, model_path: Optional[str] = None):
        """
              for training mi lowerbound only.

              Args:
                  model_path:
                      (Optional) Local path to model if model to train has been instantiated from a local path
                      If present, we will try reloading the optimizer/scheduler states from there.
              """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                    self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_mi_optimizers(num_training_steps=t_total)

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            self.mi_estimator = torch.nn.DataParallel(self.mi_estimator)


        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )
            self.mi_estimator = torch.nn.parallel.DistributedDataParallel(
                self.mi_estimator,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )


        # Train!
        if is_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        start_time = time.time()
        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        if self.mi_estimator:
            self.mi_estimator.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_process_zero()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_process_zero())

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # print(f'inputs: {inputs}')
                full_loss, loss_dict = self._adv_training_step(model, inputs, optimizer)
                tr_loss += full_loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    if self.mi_estimator:
                        self.mi_estimator.zero_grad()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logs.update(loss_dict)
                        logging_loss = tr_loss

                        self._log(logs)


                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = tr_loss

        self.is_in_train = False

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, metrics)

    # infobert
    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            if self.mi_estimator:
                self.mi_estimator = torch.nn.DataParallel(self.mi_estimator)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )
            if self.mi_estimator:
                self.mi_estimator = torch.nn.parallel.DistributedDataParallel(
                    self.mi_estimator,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                    find_unused_parameters=True,
                )


        # Train!
        if is_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        num_train_samples = self.args.max_steps * total_train_batch_size
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        start_time = time.time()
        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )
                if os.path.isfile(os.path.join(model_path, "eval_hist.bin")):
                    self.eval_hist = torch.load(os.path.join(model_path, "eval_hist.bin"))

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        if self.mi_estimator:
            self.mi_estimator.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_process_zero()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_process_zero())

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                full_loss, loss_dict = self._adv_training_step(model, inputs, optimizer)
                tr_loss += full_loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    if self.mi_estimator:
                        self.mi_estimator.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ): 
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logs.update(loss_dict)
                        logging_loss = tr_loss

                        self._log(logs)

                        if self.args.evaluate_during_training:
                            if self.global_step >= t_total // 3:   # train first epochs without evaluation
                                self.eval_hist += self.evaluate(),
                                self.eval_hist[-1]['step'] = self.global_step

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = tr_loss
        self.is_in_train = False

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, metrics)

    # infobert
    def get_seq_len(self, inputs):
        # print(inputs['attention_mask'].shape)  # [bs, max_seq_len]
        # print(inputs)
        # print(inputs['input_ids'][0])
        # print(torch.sum(inputs['input_ids'][0] != 0))
        lengths = torch.sum(inputs['attention_mask'], dim=-1)
        # print(lengths[0])
        # print(inputs['input_ids'][0])
        # print(torch.sum(inputs['input_ids'][0] != 0))
        # print(lengths.shape)  # [bs]
        return lengths.detach().cpu().numpy()

    def _train_mi_upper_estimator(self, outputs, inputs=None):
        # pos. 3 instead of 2 for BERTforqa
        hidden_states = outputs[3]  # need to set config.output_hidden = True
        # hidden_states is tuple of length 13
        # print(f'hidden states size: {hidden_states[-1].shape}') # 32 x 128 x 768
        last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768

        sentence_embedding = last_hidden[:, 0]  # batch x 768   
        if self.mi_upper_estimator.version == 0:
            embedding_layer = torch.reshape(embedding_layer, [embedding_layer.shape[0], -1])
            return self.mi_upper_estimator.update(embedding_layer, sentence_embedding)
        elif self.mi_upper_estimator.version == 1:
            return self.mi_upper_estimator.update(embedding_layer, last_hidden)
        elif self.mi_upper_estimator.version == 2:
            return self.mi_upper_estimator.update(embedding_layer, sentence_embedding)
        elif self.mi_upper_estimator.version == 3:
            embeddings = []
            lengths = self.get_seq_len(inputs)
            for i, length in enumerate(lengths):
                embeddings.append(embedding_layer[i, :length])
            embeddings = torch.cat(embeddings)  # [-1, 768]
            return self.mi_upper_estimator.update(embedding_layer, embeddings)

    def _get_local_robust_feature_regularizer(self, outputs, local_robust_features):
        # for bertforqa, change to 3 instead of 2
        hidden_states = outputs[3]  # need to set config.output_hidden = True
        last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
        sentence_embeddings = last_hidden[:, 0]  # batch x 768
        local_embeddings = []
        global_embeddings = []
        for i, local_robust_feature in enumerate(local_robust_features):
            for local in local_robust_feature:
                local_embeddings.append(embedding_layer[i, local])
                global_embeddings.append(sentence_embeddings[i])

        lower_bounds = []
        from sklearn.utils import shuffle
        local_embeddings, global_embeddings = shuffle(local_embeddings, global_embeddings, random_state=self.args.seed)
        for i in range(0, len(local_embeddings), self.args.train_batch_size):
            local_batch = torch.stack(local_embeddings[i: i + self.args.train_batch_size])
            global_batch = torch.stack(global_embeddings[i: i + self.args.train_batch_size])
            lower_bounds += self.mi_estimator(local_batch, global_batch),
        return -torch.stack(lower_bounds).mean()

    def _eval_mi_upper_estimator(self, outputs, inputs=None):
        hidden_states = outputs[2]  # need to set config.output_hidden = True
        last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
        sentence_embedding = last_hidden[:, 0]  # batch x 768
        if self.mi_upper_estimator.version == 0:
            embedding_layer = torch.reshape(embedding_layer, [embedding_layer.shape[0], -1])
            return self.mi_upper_estimator.mi_est(embedding_layer, sentence_embedding)
        elif self.mi_upper_estimator.version == 1:
            return self.mi_upper_estimator.mi_est(embedding_layer, last_hidden)
        elif self.mi_upper_estimator.version == 2:
            return self.mi_upper_estimator.mi_est(embedding_layer, sentence_embedding)
        elif self.mi_upper_estimator.version == 3:
            embeddings = []
            lengths = self.get_seq_len(inputs)
            for i, length in enumerate(lengths):
                embeddings.append(embedding_layer[i, :length])
            embeddings = torch.cat(embeddings)  # [-1, 768]
            return self.mi_upper_estimator.mi_est(embedding_layer, embeddings)
    
    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ):
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)
  
        outputs = model(**inputs)

        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        if self.mi_upper_estimator:
            upper_bound = self._train_mi_upper_estimator(outputs, inputs)
            loss += upper_bound

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.mi_upper_estimator:
                upper_bound = upper_bound.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if self.mi_estimator:
            loss_dict = {"task_loss": loss.item(), "upper_bound": upper_bound.item()}
            return loss.item(), loss_dict
        return loss.item()

    def feature_ranking(self, grad, cl=0.5, ch=0.9):
        """
        :param grad: [seq_len, hidden_size]
        :param cl: ranking lower threshold
        :param ch: ranking upper threshold (0 <= cl <= ch <= 1)
        :return: local robust word posids whose perturbation is between cl and ch (list of one-dim tensor)
        """
        n = len(grad)
        import math
        lower = math.ceil(n * cl)
        upper = math.ceil(n * ch)
        norm = torch.norm(grad, dim=1)  # [seq_len]
        _, ind = torch.sort(norm)
        res = []
        for i in range(lower, upper):
            res += ind[i].item(),
        return res

    def convert_ids_to_string(self, ids):
        tokens = []
        for id in ids:
            tokens += self.tokenizer._convert_id_to_token(id.item()),
        return self.tokenizer.convert_tokens_to_string(tokens)


    def local_robust_feature_selection(self, inputs, grad, input_ids=None):
        """
        :param input_ids: for visualization, print out the local robust features
        :return: list of list of local robust feature posid, non robust feature posid
        """
        grads = []
        lengths = self.get_seq_len(inputs)
        for i, length in enumerate(lengths):
            grads.append(grad[i, :length])
        indices = []
        nonrobust_indices = []
        for i, grad in enumerate(grads):
            indices.append(self.feature_ranking(grad, self.args.cl, self.args.ch))
            nonrobust_indices.append([x for x in range(lengths[i]) if x not in indices])
            # for visualization
            # idxs = indices[-1]
            # idxs = input_ids[i, idxs]
            # text = self.convert_ids_to_string(input_ids[i, :lengths[i]])
            # features = self.convert_ids_to_string(idxs)
        return indices, nonrobust_indices

    def _adv_training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ):
        model.train()
        if self.mi_estimator:
            self.mi_estimator.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        # ============================ Code for adversarial training=============
        tr_loss, upperbound_loss, lowerbound_loss = 0.0, 0.0, 0.0
        input_ids = inputs.pop('input_ids')

        # initialize delta
        if hasattr(model, 'module'):
            if hasattr(model.module, 'bert'):
                embeddings = model.module.bert.embeddings.word_embeddings
            elif hasattr(model.module, 'roberta'):
                embeddings = model.module.roberta.embeddings.word_embeddings
            clear_mask = model.module.clear_mask
        else:
            if hasattr(model, 'bert'):
                embeddings = model.bert.embeddings.word_embeddings
            elif hasattr(model, 'roberta'):
                embeddings = model.roberta.embeddings.word_embeddings
            clear_mask = model.clear_mask

        embeds_init = embeddings(input_ids)

        if self.args.adv_init_mag > 0:
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            # check the shape of the mask here..

            if self.args.norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.args.norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.args.adv_init_mag,
                                                               self.args.adv_init_mag) * input_mask.unsqueeze(2)

        else:
            delta = torch.zeros_like(embeds_init)

        # the main loop
        clear_mask()
        for astep in range(self.args.adv_steps):
            # (0) forward
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init

            outputs = model(**inputs)
            # (loss, start, end, hidden, attention) = outputs
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            # (1) backward
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss = loss / self.args.adv_steps

            tr_loss += loss.item()

            if self.mi_upper_estimator:
                upper_bound = self._train_mi_upper_estimator(outputs, inputs) / self.args.adv_steps
                loss += upper_bound
                upperbound_loss += upper_bound.item()

            if self.args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()
            if self.mi_estimator:
                local_robust_features, _ = self.local_robust_feature_selection(inputs, delta_grad, input_ids)
                lower_bound = self._get_local_robust_feature_regularizer(outputs, local_robust_features) * \
                              self.args.alpha / self.args.adv_steps
                lower_bound.backward()
                lowerbound_loss += lower_bound.item()

            if astep == self.args.adv_steps - 1:  ## if no freelb, set astep = 1, adv_init=0
                # further updates on delta
                break

            # (3) update and clip
            if self.args.norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.args.adv_max_norm).to(embeds_init)
                    reweights = (self.args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.args.norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.args.adv_max_norm, self.args.adv_max_norm).detach()
            else:
                print("Norm type {} not specified.".format(self.args.norm_type))
                exit()

            embeds_init = embeddings(input_ids)
        clear_mask()

        loss_dict = {"task_loss": tr_loss}
        if self.mi_upper_estimator:
            loss_dict.update({"upper_bound": upperbound_loss})
        if self.mi_estimator:
            loss_dict.update({"lower_bound": lowerbound_loss})
        return tr_loss, loss_dict   


    def _prediction_mi_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()
        if self.mi_estimator:
            self.mi_estimator.eval()

        if self.mi_upper_estimator:
            mi_info = []

        if self.mi_estimator:
            mi_info_lower = []
            mi_info_nonrobust_lower = []

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

            # initialize delta
        if hasattr(model, 'module'):
            if hasattr(model.module, 'bert'):
                embeddings = model.module.bert.embeddings.word_embeddings
            elif hasattr(model.module, 'roberta'):
                embeddings = model.module.roberta.embeddings.word_embeddings
            clear_mask = model.module.clear_mask
        else:
            if hasattr(model, 'bert'):
                embeddings = model.bert.embeddings.word_embeddings
            elif hasattr(model, 'roberta'):
                embeddings = model.roberta.embeddings.word_embeddings
            clear_mask = model.clear_mask

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            clear_mask()
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if self.mi_upper_estimator:
                    mi_info += self._eval_mi_upper_estimator(outputs, inputs),


            if has_labels:
                input_ids = inputs.pop('input_ids')
                embeds_init = embeddings(input_ids)

                if self.args.adv_init_mag > 0:

                    input_mask = inputs['attention_mask'].to(embeds_init)
                    input_lengths = torch.sum(input_mask, 1)
                    # check the shape of the mask here..

                    if self.args.norm_type == "l2":
                        delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                        dims = input_lengths * embeds_init.size(-1)
                        mag = self.args.adv_init_mag / torch.sqrt(dims)
                        delta = (delta * mag.view(-1, 1, 1)).detach()
                    elif self.args.norm_type == "linf":
                        delta = torch.zeros_like(embeds_init).uniform_(-self.args.adv_init_mag,
                                                                       self.args.adv_init_mag) * input_mask.unsqueeze(2)

                else:
                    delta = torch.zeros_like(embeds_init)

                # the main loop
                for astep in range(1):
                    # (0) forward
                    delta.requires_grad_()
                    inputs['inputs_embeds'] = delta + embeds_init

                    clear_mask()
                    adv_outputs = model(**inputs)
                    loss = adv_outputs[0]  # model outputs are always tuple in transformers (see doc)
                    # (1) backward
                    if self.args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    loss = loss / self.args.adv_steps

                    loss.backward()

                    # (2) get gradient on delta
                    delta_grad = delta.grad.clone().detach()

                    if self.mi_estimator:
                        with torch.no_grad():
                            local_robust_features, nonrobust_features = \
                                self.local_robust_feature_selection(inputs, delta_grad, input_ids)
                            lower_bound = self._get_local_robust_feature_regularizer(outputs, local_robust_features) \
                                          / self.args.adv_steps
                            mi_info_lower += lower_bound,
                            nonrobust_lower_bound = self._get_local_robust_feature_regularizer(outputs, nonrobust_features) \
                                          / self.args.adv_steps
                            mi_info_nonrobust_lower += nonrobust_lower_bound,

                        model.zero_grad()
                        self.mi_estimator.zero_grad()

                    if astep == self.args.adv_steps - 1:
                        # further updates on delta
                        break

                    # (3) update and clip
                    if self.args.norm_type == "l2":
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                        if self.args.adv_max_norm > 0:
                            delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                            exceed_mask = (delta_norm > self.args.adv_max_norm).to(embeds_init)
                            reweights = (self.args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                                     1)
                            delta = (delta * reweights).detach()
                    elif self.args.norm_type == "linf":
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                        if self.args.adv_max_norm > 0:
                            delta = torch.clamp(delta, -self.args.adv_max_norm, self.args.adv_max_norm).detach()
                    else:
                        print("Norm type {} not specified.".format(self.args.norm_type))
                        exit()

                    embeds_init = embeddings(input_ids)

            if has_labels:
                step_eval_loss, logits = outputs[:2]
                eval_losses += [step_eval_loss.mean().item()]
            else:
                logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        clear_mask()

        if self.mi_upper_estimator:
            mi_info = torch.stack(mi_info)
            # print(mi_info.shape)

        if self.mi_estimator:
            mi_info_lower = torch.stack(mi_info_lower)
            mi_info_nonrobust_lower = torch.stack(mi_info_nonrobust_lower)
            # print(mi_info_lower.shape)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
            if self.mi_upper_estimator:
                mi_info = distributed_concat(mi_info, num_total_examples=self.num_examples(dataloader))
            if self.mi_estimator:
                mi_info_lower = distributed_concat(mi_info_lower, num_total_examples=self.num_examples(dataloader))
                mi_info_nonrobust_lower = distributed_concat(mi_info_nonrobust_lower, num_total_examples=self.num_examples(dataloader))
        elif is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            print(f'metrics: {metrics}')
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)
        if self.mi_upper_estimator:
            metrics['mi_info'] = mi_info.mean().item()
        if self.mi_estimator:
            metrics['mi_info_lower'] = mi_info_lower.mean().item()
            metrics['mi_info_lower_nonrobust'] = mi_info_nonrobust_lower.mean().item()

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)


    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only
        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        start_preds: torch.Tensor = None
        end_preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()
        if self.mi_estimator:
            self.mi_estimator.eval()

        if self.mi_upper_estimator:
            mi_info = []

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if hasattr(model, 'module'):
            clear_mask = model.module.clear_mask
        else:
            clear_mask = model.clear_mask

        all_start_logits = []
        all_end_logits = []
        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            clear_mask()
            with torch.no_grad():
                outputs = model(**inputs)
                if self.mi_upper_estimator:
                    mi_info += self._eval_mi_upper_estimator(outputs, inputs),
                if has_labels:
                    step_eval_loss = outputs[0].tolist()
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    # print(f'outputs: {outputs}')
                    start_logits, end_logits = outputs
                    all_start_logits.append(start_logits)
                    all_end_logits.append(end_logits)


            if not prediction_loss_only:
                if start_preds is None:
                    start_preds = start_logits.detach()
                    end_preds = end_logits.detach()
                else:
                    start_preds = torch.cat((start_preds, start_logits.detach()), dim=0)
                    end_preds = torch.cat((end_preds, end_logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        clear_mask()

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if start_preds is not None:
                start_preds = distributed_concat(start_preds, num_total_examples=self.num_examples(dataloader))
                end_preds = distributed_concat(end_preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        # elif is_tpu_available():
        #     # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
        #     if preds is not None:
        #         preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
        #     if label_ids is not None:
        #         label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if start_preds is not None:
            start_preds = start_preds.cpu().numpy()
            end_preds = end_preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=(start_preds, end_preds), label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)
        if self.mi_upper_estimator:
            metrics['mi_info'] = torch.tensor(mi_info).mean().item()

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=(start_preds, end_preds), label_ids=label_ids, metrics=metrics)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

############# original code
    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self._prediction_loop( # from prediction_loop
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                # ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self._prediction_loop(
                test_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                # ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        eval_preds = self.post_process_function(test_examples, test_dataset, output.predictions)
        metrics = self.compute_metrics(eval_preds)

        return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)