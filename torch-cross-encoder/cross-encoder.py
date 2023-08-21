import torch
from transformers import AutoModel, AutoTokenizer, RobertaConfig

class Crossencoder(torch.nn.Module):
    def __init__(
            self,
            hidden_dim:int=768,
            model_path:str="vinai/phobert-base-v2",
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.dense = torch.nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim

    def forward(
            self,
            inputs,
    ):
        input_ids, inputs_segments, inputs_atten_mask = inputs

        if len(input_ids.shape) >= 3:
            _, sample_per_ques, max_length = input_ids.shape
            input_ids         = input_ids.view(-1, max_length)
            inputs_segments   = inputs_segments.view(-1, max_length)
            inputs_atten_mask = inputs_atten_mask.view(-1, max_length)

            rep_vector = self.encoder(input_ids = input_ids, # shape = (batch*sample_per_question, max_length)
                                    attention_mask = inputs_atten_mask,
                                    token_type_ids = inputs_segments,
                                    ).pooler_output # shape = (batch_size*sample_per_question, hidden_size)
            rep_vector = rep_vector.view(-1, sample_per_ques, self.hidden_dim)
        else:
            rep_vector = self.encoder(input_ids = input_ids, # shape = (batch*sample_per_question, max_length)
                                    attention_mask = inputs_atten_mask,
                                    token_type_ids = inputs_segments,
                                    ).pooler_output # shape = (batch_size*sample_per_question, hidden_size)

        output = self.dense(rep_vector) # output = (batch_size, sample_per_question, 2)
        return output

    def cal_loss(self, score_vector, postitive_idx_per_question): # postitive_idx_per_question = (batch_size, sample_per_question)
        #score_vector = torch.squeeze(score_vector) # shape = (batch, sample_per_question)
        logit = torch.nn.LogSoftmax(dim = -1)(score_vector) # shape = (batch, sample_per_question, 2)
        #loss = torch.mul(postitive_idx_per_question, logit).mean()
        logit = logit.view(-1, 2)
        postitive_idx_per_question = postitive_idx_per_question.view(-1)
        loss = torch.nn.NLLLoss()(logit, postitive_idx_per_question)
        return loss