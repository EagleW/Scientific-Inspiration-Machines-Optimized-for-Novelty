# based on https://github.com/intfloat/SimKGC
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from dataclasses import dataclass
import torch
from utils_ import accuracy
from typing import List

@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor
    loss: torch.tensor
    acc: List


class ContrastiveDualEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model)
        self.bert = AutoModel.from_pretrained(args.model)

        del self.bert.pooler
        self.bert.pooler = None
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        self.add_margin = args.additive_margin
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log())
        self.num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(self.num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.not_first = False
        self.criterion = nn.CrossEntropyLoss()

    def _encode(self, token_ids, mask, token_type_ids=None):
        if 'roberta' in self.args.model.lower() or token_type_ids is None:
            outputs = self.bert(input_ids=token_ids,
                            attention_mask=mask,
                            return_dict=True)
        else:
            outputs = self.bert(input_ids=token_ids,
                            attention_mask=mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, 
                    tail_token_ids=None, tail_mask=None,
                    head_token_ids=None, head_mask=None,
                    hr_token_type_ids=None, only_ent_embedding=False,
                    neg_ids=None, neg_mask=None,
                    only_hr_embedding=False,
                    **kwargs):
        if only_ent_embedding:
            return self.predict_ent_embedding(hr_token_ids=hr_token_ids,
                                              hr_mask=hr_mask)

        hr_vector = self._encode(
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
        if only_hr_embedding:
            return {'hr_vector': hr_vector}

        tail_vector = self._encode(
                                   token_ids=tail_token_ids,
                                   mask=tail_mask)

        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            head_vector = self._encode(
                                    token_ids=head_token_ids,
                                    mask=head_mask)
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)

            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1, keepdim=True) 
            logits = torch.cat([logits, self_neg_logits], dim=-1)

            # context_vector = self._encode(
            #                         token_ids=neg_ids,
            #                         mask=neg_mask).view(batch_size, -1, head_vector.size(-1))
            # self_context_logits = torch.sum(hr_vector.unsqueeze(1) * context_vector, dim=-1) * self.args.pre_batch_weight
            # logits = torch.cat([logits, self_context_logits], dim=-1)
        

            if self.pre_batch > 0:
                if self.not_first:
                    pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t()) * self.args.pre_batch_weight
                    logits = torch.cat([logits, pre_batch_logits], dim=-1)
                adding_size = min(self.num_pre_batch_vectors - self.offset, batch_size)
                self.pre_batch_vectors[self.offset:(self.offset + adding_size)] = tail_vector[:adding_size].data.clone()
                tmp = self.offset + batch_size - self.num_pre_batch_vectors
                if tmp > 0:
                    self.pre_batch_vectors[:tmp] = tail_vector[:tmp].data.clone()
                    self.offset = tmp
                else:
                    self.offset = self.offset + batch_size
                self.not_first = True
            

        logits *= self.log_inv_t.exp()
        # head + relation -> tail
        loss = self.criterion(logits, labels)
        # tail -> head + relation
        loss += self.criterion(logits[:, :batch_size].t(), labels)
        accs = accuracy(logits, labels, topk=(1, 10))

        
        return ModelOutput(logits= logits,
                labels= labels,
                inv_t= self.log_inv_t.detach().exp(),
                hr_vector= hr_vector.detach(),
                tail_vector= tail_vector.detach(),
                loss=loss,
                acc=accs
                )

    @torch.no_grad()
    def predict_ent_embedding(self, hr_token_ids, hr_mask, **kwargs):
        ent_vectors = self._encode(
                                   token_ids=hr_token_ids,
                                   mask=hr_mask)
        return {'ent_vectors': ent_vectors.detach()}





def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector