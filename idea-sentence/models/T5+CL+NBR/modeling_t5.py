from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import logging

from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5ForConditionalGeneration,
    T5Config,
)
import copy
import warnings
from torch.utils.checkpoint import checkpoint


import torch.nn.functional as F
logger = logging.get_logger(__name__)


class ContrastiveSeq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None



class ContrastiveHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        inner_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, 1)

    def forward(self, hidden_states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.avg_pool(hidden_states, masks)
        hidden_states = torch.sigmoid(hidden_states)
        return hidden_states

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class ContrastiveT5ForConditionalGeneration(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.contrastive_head = ContrastiveHead(config.d_model, config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,

        neg_ids: Optional[torch.LongTensor] = None,
        neg_num_total: Optional[int]=1,
        valid: Optional[bool] = False,
        ppl: Optional[bool] = False,

        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        masked_lm_loss = None
        contrastive_loss = None
        loss = None
        if labels is not None:
            if ppl:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum')
                ppl_list = []
                for k in range(labels.size(0)):
                    masked_lm_loss = loss_fct(lm_logits[k,:,:], labels[k,:])
                    # prob = torch.exp(-masked_lm_loss)
                    prob = -masked_lm_loss
                    ppl_list.append(prob.item())
                return ppl_list
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            if not valid:
                contrastive_loss = self.contrastive(
                    decoder_outputs,
                    labels,
                    attention_mask,
                    neg_ids,
                    neg_num_total,
                    encoder_outputs,
                    decoder_attention_mask,
                    past_key_values,
                    decoder_inputs_embeds,
                    decoder_head_mask,
                    output_attentions,
                    cross_attn_head_mask,
                    output_hidden_states,
                    return_dict,
                    use_cache)
                loss = masked_lm_loss + contrastive_loss
            else:
                loss = masked_lm_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,masked_lm_loss,contrastive_loss,) + output) if loss is not None else output

        return ContrastiveSeq2SeqLMOutput(
            loss=loss,
            mlm_loss=masked_lm_loss,
            cl_loss=contrastive_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def contrastive(
        self,
        decoder_outputs,
        labels,
        attention_mask,
        neg_ids,
        expand_size,
        encoder_outputs,
        decoder_attention_mask,
        past_key_values,
        decoder_inputs_embeds,
        decoder_head_mask,
        output_attentions,
        cross_attn_head_mask,
        output_hidden_states,
        return_dict,
        use_cache
    ):
        pos_label_mask = labels != -100
        pos_emb = self.contrastive_head(decoder_outputs.last_hidden_state, pos_label_mask)


        decoder_input_ids = self._shift_right(neg_ids)
        bs = labels.size(0)

        expanded_return_idx = (
            torch.arange(attention_mask.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(attention_mask.device)
        )

        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )

        attention_mask = attention_mask.index_select(0, expanded_return_idx).to(attention_mask.device)


        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        neg_label_mask = neg_ids != self.config.pad_token_id
        neg_emb = self.contrastive_head(decoder_outputs.last_hidden_state, neg_label_mask).view(bs, expand_size)


        loss_fct = CrossEntropyLoss()
        all_logit = torch.cat([pos_emb,neg_emb], dim=1)
        # print(bs)
        l = torch.zeros([bs], dtype=torch.long, device=neg_emb.device)
        # l = torch.ones_like(neg_emb, dtype=torch.long, device=neg_emb.device)
        cl_loss = loss_fct(all_logit, l)
        # loss_cl = MarginRankingLoss(0.5)
        # pos_emb = pos_emb.index_select(
        #     0, expanded_return_idx.to(pos_emb.device)
        # )
        # cl_loss = loss_cl(pos_emb, neg_emb, l)
        return cl_loss