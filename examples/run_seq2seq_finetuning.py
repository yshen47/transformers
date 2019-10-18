# coding=utf-8
# Copyright 2018 The Microsoft Reseach team and The HuggingFace Inc. team.
# Copyright (c) 2018 Microsoft and The HuggingFace Inc.  All rights reserved.
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
""" Finetuning seq2seq models for sequence generation."""

import argparse
from collections import deque
import logging
import os
import pickle
import random
import sys

import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler

from transformers import AutoTokenizer, PreTrainedSeq2seq, Model2Model, WarmupLinearSchedule

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# ------------
# Load dataset
# ------------


class TextDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, tokenizer, prefix="train", data_dir="", block_size=512):
        assert os.path.isdir(data_dir)

        # Load features that have already been computed if present
        cached_features_file = os.path.join(
            data_dir, "cached_lm_{}_{}".format(block_size, prefix)
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as source:
                examples = pickle.load(source)
                self.source_examples = examples["source"]
                self.target_examples = examples["target"]
                return

        logger.info("Creating features from dataset at %s", data_dir)
        self.source_examples = []
        self.target_examples = []
        datasets = ["cnn", "dailymail"]
        for dataset in datasets:
            path_to_stories = os.path.join(data_dir, dataset, "stories")
            assert os.path.isdir(path_to_stories)

            story_filenames_list = os.listdir(path_to_stories)
            for story_filename in story_filenames_list:
                path_to_story = os.path.join(path_to_stories, story_filename)
                if not os.path.isfile(path_to_story):
                    continue

                with open(path_to_story, encoding="utf-8") as source:
                    try:
                        raw_story = source.read()
                        story, summary = process_story(raw_story)
                    except IndexError:  # skip ill-formed stories
                        continue

                story = tokenizer.encode(story)
                story_seq = _fit_to_block_size(story, block_size)
                self.source_examples.append(story_seq)

                summary = tokenizer.encode(summary)
                summary_seq = _fit_to_block_size(summary, block_size)
                self.target_examples.append(summary_seq)

        logger.info("Saving features into cache file %s", cached_features_file)
        with open(cached_features_file, "wb") as sink:
            pickle.dump(
                {"source": self.source_examples, "target": self.target_examples},
                sink,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def __len__(self):
        return len(self.source_examples)

    def __getitem__(self, items):
        return (
            torch.tensor(self.source_examples[items]),
            torch.tensor(self.target_examples[items]),
        )


def process_story(raw_story):
    """ Extract the story and summary from a story file.

    Attributes:
        raw_story (str): content of the story file as an utf-8 encoded string.

    Raises:
        IndexError: If the stoy is empty or contains no highlights.
    """
    file_lines = list(
        filter(lambda x: len(x) != 0, [line.strip() for line in raw_story.split("\n")])
    )

    # for some unknown reason some lines miss a period, add it
    file_lines = [_add_missing_period(line) for line in file_lines]

    # gather article lines
    story_lines = []
    lines = deque(file_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith("@highlight"):
                break
            story_lines.append(element)
        except IndexError as ie:  # if "@highlight" absent from file
            raise ie

    # gather summary lines
    highlights_lines = list(filter(lambda t: not t.startswith("@highlight"), lines))

    # join the lines
    story = " ".join(story_lines)
    summary = " ".join(highlights_lines)

    return story, summary


def _add_missing_period(line):
    END_TOKENS = [".", "!", "?", "...", "'", "`", '"', u"\u2019", u"\u2019", ")"]
    if line.startswith("@highlight"):
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + "."


def _fit_to_block_size(sequence, block_size):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter than the block size we pad it with -1 ids
    which correspond to padding tokens.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([0] * (block_size - len(sequence)))
        return sequence


def mask_padding_tokens(sequence):
    """ Replace the padding token with -1 values """
    padded = sequence.clone()
    padded[padded == 0] = -1
    return padded


def load_and_cache_examples(args, tokenizer):
    dataset = TextDataset(tokenizer, data_dir=args.data_dir)
    return dataset


# ------------
# Train
# ------------


def train(args, train_dataset, model, tokenizer):
    """ Fine-tune the pretrained model on the corpus. """

    # Prepare the data loading
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            t_total
            // ( len(train_dataloader) // args.gradient_accumulation_steps + 1))
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare the optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=t_total
    )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
        # * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=True)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            source, target = batch
            labels = mask_padding_tokens(target)
            source = source.to(args.device)
            target = target.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(
                source,
                target,
                decoder_attention_mask=labels,
                decoder_lm_labels=labels,
            )

            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Optional parameters
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--do_train", type=bool, default=False, help="Whether to run training."
    )
    parser.add_argument(
        "--do_overwrite_output",
        type=bool,
        default=False,
        help="Whether to overwrite the output dir.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint to initialize the encoder and decoder's weights with.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The decoder architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--to_cpu", default=False, type=bool, help="Whether to force training on CPU."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    args = parser.parse_args()

    if args.model_type != "bert":
        raise ValueError(
            "Only the BERT architecture is currently supported for seq2seq."
        )
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overwrite.".format(
                args.output_dir
            )
        )

    # Set up training device
    if args.to_cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.device = torch.device("cpu")
        args.n_gpu = torch.cuda.device_count()

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = Model2Model.from_pretrained(args.model_name_or_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        0,
        args.device,
        args.n_gpu,
        False,
        False,
    )

    logger.info("Training/evaluation parameters %s", args)

    # Train the model
    model.to(args.device)
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_arguments.bin"))

    # Evaluate the model
    results = {}
    if args.do_eval:
        checkpoints = []
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            encoder_checkpoint = os.path.join(checkpoint, "encoder")
            decoder_checkpoint = os.path.join(checkpoint, "decoder")
            model = PreTrainedSeq2seq.from_pretrained(encoder_checkpoint, decoder_checkpoint)
            model.to(args.device)
            results = "placeholder"

    return results


if __name__ == "__main__":
    main()
