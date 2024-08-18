import torch
import transformers
from transformers import BertPreTrainedModel
import torch.nn as nn
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers import XLMRobertaModel
from typing import Optional


class RobertaSeriesConfig(XLMRobertaConfig):
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2,project_dim=768,pooler_fn='cls',learn_encoder=False, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        

class RobertaSeriesModelWithTransformation(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    base_model_prefix = 'roberta'
    config_class= RobertaSeriesConfig
    def __init__(self, config):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        self.post_init()
        
    def get_text_embeds(self,bert_embeds,clip_embeds):
        return self.merge_head(torch.cat((bert_embeds,clip_embeds)))

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, input_ids: Optional[torch.Tensor] = None) :
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.int64)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        projection_state = self.transformation(outputs.last_hidden_state)
        
        return (projection_state,)