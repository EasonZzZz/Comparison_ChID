import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput
from typing import Optional, Union, Tuple


def batched_index_select(tensor, indices):
    view_shape = list(tensor.shape)
    view_shape[1] = -1
    return tensor.gather(1, indices.view(*view_shape))


def bert_forward(
        model,
        cls,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        candidates: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]
    prediction_scores = cls(sequence_output)

    masked_lm_loss = None
    candidate_final_scores = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        candidate_prediction_scores = torch.masked_select(prediction_scores, candidate_mask.unsqueeze(-1)).reshape(
            -1, prediction_scores.shape[-1], 1)
        candidate_indices = candidates.transpose(-1, -2).reshape(-1, candidates.shape[1])
        candidate_logits = batched_index_select(candidate_prediction_scores, candidate_indices).squeeze(-1).reshape(
            prediction_scores.shape[0], 4, -1).transpose(-1, -2)
        candidate_labels = labels.reshape(labels.shape[0], 1).repeat(1, 4)
        candidate_final_scores = torch.sum(F.log_softmax(candidate_logits, dim=-2), dim=-1)

        masked_lm_loss = loss_fct(candidate_logits, candidate_labels)

    if not return_dict:
        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return MaskedLMOutput(
        loss=masked_lm_loss,
        logits=candidate_final_scores if candidate_final_scores is not None else prediction_scores,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
