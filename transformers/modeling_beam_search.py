# coding=utf-8
""" A general wrapper around models with LM heads to
perform beam search.
"""

import torch


class ModelWithBeamSearch(object):
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

        # forward pass on the encoder
        encoder_outputs = self.encoder(input_ids, kwargs_encoder)

        # initialize variables
        batch_size, _ = input_ids.size()
        topk_log_probabilities = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1)
        ).repeat(batch_size)

        results = {}
        results["predictions"] = [[] for _ in batch_size]
        results["scores"] = [[] for _ in batch_size]

        # For each batch we need to make `beam_size` predictions at each step.
        # We thus need to repeat the encoder hidden states `beam_size` times.
        kwargs_decoder["encoder_hidden_states"] = tile(
            encoder_outputs, self.beam_size, dim=0
        )
        beam_offset = torch.arange(
            0, batch_size * self.beam_size, step=self.beam_size, dtype=torch.long
        )
        hypotheses = [[] for _ in range(batch_size)]
        batch_offset = torch.arange(batch_size, dtype=torch.long)
        growing_sequence = torch.full(
            (self.batch_size * self.beam_size, 1), self.start_token, dtype=torch.long
        )

        for step in range(self.max_length):
            decoder_input = growing_sequence[:, -1]
            outputs = self.decoder(decoder_input, kwargs_decoder)
            log_probabilities = outputs[1]
            vocab_size = log_probabilities.size(-1)

            # Multiply each beam probability with the probability of the
            # next token (conditioned on the words in the beam).
            log_probabilities += topk_log_probabilities.view()

            # if the beam has not attained the minimum required length we
            # make the end token arbitrarily unlikely.
            if step < self.min_length:
                log_probabilities[self.end_token] = -1e20

            # TODO: remove trigrams

            # Find the `beam_size` (previous_beam + token) combinations with
            # the highest score
            log_probabilities = log_probabilities.reshape(
                -1, self.beam_size * vocab_size
            )
            topk_log_probabilities, topk_ids = log_probabilities.topk(
                self.beam_size, dim=-1
            )

            # Length penalty. The +1 accounts for the [EOS] token
            # that will be added if the beam ends.
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** self.alpha
            topk_scores = log_probabilities / length_penalty

            # Retrieve the corresponding respective beam and token id
            # topk_token_ids[i] will be added to topk_beam_ids[i]
            topk_beam_ids = topk_ids.div(self.beam_size)
            topk_token_ids = topk_ids.fmod(vocab_size)

            # Retrieve the index of the concerned beams in the original
            # log_probabilities tensor
            batch_index = (
                topk_beam_ids + beam_offset[: topk_beam_ids.size(0).unsqueeze(1)]
            )
            select_indices = batch_index.view(-1)

            # Append the last predictions
            growing_sequence = torch.cat(
                [
                    growing_sequence.index_select(0, select_indices),
                    topk_token_ids.view(-1, 1),
                ],
                -1,
            )

            # Check if any of the beam searches has ended
            is_finished = topk_token_ids.eq(self.end_token)
            if step + 1 == self.max_length:
                is_finished.fill_(1)
            is_top_beam_finished = is_finished[:, 0].eq(1)

            # Save the finished searches
            if is_finished.any():
                predictions = growing_sequence.view(-1, self.beam_size, growing_sequence.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if is_top_beam_finished[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    # If the batch reached the end, save the best hypotheses.
                    if is_top_beam_finished[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True
                        )
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = is_top_beam_finished.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probabilities = topk_log_probabilities.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                growing_sequence = predictions.index_select(0, non_finished).view(
                    -1, growing_sequence.size(-1)
                )

            # Re-order the state for the next pass
            select_indices = batch_index.view(-1)
            kwargs_decoder["encoder_hidden_states"] = kwargs_decoder[
                "encoder_hidden_states"
            ].index_select(0, select_indices)

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
