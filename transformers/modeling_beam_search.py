# coding=utf-8
# Copyright (c) 2019 Yang Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
A general wrapper around models with LM heads to generate sequences
using beam search.
"""
import torch
from torch import nn


class ModelWithBeamSearch(nn.Module):
    def __init__(
        self,
        model,
        beam_size,
        start_token,
        end_token,
        pad_token,
        min_length,
        max_length,
        alpha,
        block_trigram=True,
    ):
        """
        Attributes:
            mask_word_id: token id that corresponds to the mask
        """
        super(ModelWithBeamSearch, self).__init__()
        self.model = model
        self.beam_size = beam_size
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.min_length = min_length
        self.max_length = max_length
        self.alpha = alpha
        self.block_trigram = block_trigram

    def forward(self, input_ids, **kwargs):
        # Separate the encoder- and decoder- specific kwargs. A kwarg is
        # decoder-specific it the key starts with `decoder_`
        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("decoder_")
        }
        kwargs_decoder = {
            argument[len("decoder_"):]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        batch_size, _ = input_ids.size(0)

        # Variables that keep track of the status of the search
        hypotheses = [[] for _ in range(batch_size)]
        batch_offset = torch.arange(batch_size, dtype=torch.long)
        beam_offset = torch.arange(
            0,
            batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
        )
        growing_beam = torch.full(
            (batch_size * self.beam_size, 1),
            self.start_token,
            dtype=torch.long,
        )
        topk_log_probabilities = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1),
            dtype=torch.float,
        ).repeat(batch_size)

        # Forward pass on the encoder
        encoder_outputs = self.encoder(input_ids, kwargs_encoder)
        kwargs_decoder["encoder_hidden_states"] = tile(
            encoder_outputs, self.beam_size, dim=0
        )

        results = {}
        results["predictions"] = [[] for _ in batch_size]
        results["scores"] = [[] for _ in batch_size]

        for step in range(self.max_length):
            decoder_input = growing_beam[:, -1]
            outputs = self.decoder(decoder_input, kwargs_decoder)
            log_probabilities = torch.nn.functional.log_softmax(outputs[1])
            vocab_size = log_probabilities.size(-1)

            # The batch size changes as some beams finish so we define:
            _B = log_probabilities.size(0) // self.beam_size

            # Multiply each beam probability with the probability of the
            # next token (conditioned on the words in the beam).
            log_probabilities += topk_log_probabilities.view(-1, 1)

            # if the beam has not attained the minimum required length we
            # make the end token arbitrarily unlikely.
            if step < self.min_length:
                log_probabilities[self.end_token] = -1e20

            # Remove repeating tri-grams
            if(self.args.block_trigram):
                if(step + 1 > 3):
                    for i in range(_B * self.beam_size):
                        tokens = [t for t in growing_beam[i]]
                        trigrams = [(tokens[i-1], tokens[i], tokens[i+1]) for i in range(1, len(words) - 1)]
                        last_trigram = tuple(trigrams[-1])
                        if last_trigram in trigrams[:-1]:
                            log_probabilities[i] = -1e20

            # Find the `beam_size` (previous_beam + token) combinations with
            # the highest score
            topk_log_probabilities, topk_ids = log_probabilities.topk(
                log_probabilities.view(_B, self.beam_size * vocab_size),
                self.beam_size,
                dim=1
            )

            # Apply the length penalty. The +1 accounts for the [EOS] token
            # that will be added if the beam ends.
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** self.alpha
            topk_scores = topk_log_probabilities / length_penalty

            # Retrieve the corresponding respective beam and token id
            # topk_token_ids[i] will be added to topk_beam_ids[i]
            topk_beam_ids = topk_ids.div(vocab_size)
            topk_token_ids = topk_ids.fmod(vocab_size)

            # Retrieve the row index of the surviving beams in the original
            # view of the log_probabilities tensor
            surviving_beams_rows = (
                topk_beam_ids + beam_offset[:_B].view(-1, 1)
            ).view(-1)

            # Append the last predictions
            growing_beam = torch.cat(
                [
                    growing_beam.index_select(0, surviving_beams_rows),
                    topk_token_ids.view(-1, 1),
                ],
                1,
            )

            # Check if any of the beam searches has ended during this
            # growth step. Also if top beam (most probable) has ended
            # for one element of the batch.
            is_finished = topk_token_ids.eq(self.end_token)
            if step + 1 == self.max_length:
                is_finished.fill_(1)
            is_top_beam_finished = is_finished[:, 0].eq(1)

            # Save the finished searches
            if is_finished.any():
                predictions = growing_beam.view(-1, self.beam_size, growing_beam.size(1))
                for i in range(is_finished.size(0)):
                    if is_top_beam_finished[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)

                    # Store finished hypotheses for this batch.
                    b = batch_offset[i]
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, :]))

                    # If the batch reached the end, save the best hypotheses
                    # in terms of length-penalized score.
                    if is_top_beam_finished[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True
                        )
                        best_score, best_prediction = best_hyp[0]
                        results["scores"][b].append(best_score)
                        results["predictions"][b].append(best_prediction)

                non_finished = is_top_beam_finished.eq(0).nonzero().view(-1)
                if len(non_finished) == 0:
                    break

                # Remove finished batches for the next step.
                topk_log_probabilities = topk_log_probabilities.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                growing_beam = predictions.index_select(0, non_finished).view(
                    -1, growing_beam.size(-1)
                )

            # Re-order the state for the next pass
            surviving_beams_rows = surviving_beams_rows.index_select(0, non_finished)
            kwargs_decoder["encoder_hidden_states"] = kwargs_decoder[
                "encoder_hidden_states"
            ].index_select(0, surviving_beams_rows)

        return results


def tile(x, count, dim=0):
    """
    Tiles `x` along dimension `dim` `count` times.

    Example:
        >> ex = torch.tensor([1,2],[3,4])
        >> tile(ex, 2, 0)
        torch.Tensor([[1,2],[1,2],[3,4],[3,4]])
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
