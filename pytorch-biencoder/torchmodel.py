import torch
from transformers import AutoModel, AutoTokenizer, RobertaConfig

class Biencoder(torch.nn.Module):
    def __init__(
            self,
            model_path:str="vinai/phobert-base-v2",
    ):
        super().__init__()
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = RobertaConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.q_model = AutoModel.from_pretrained(model_path)#.to(device)
        self.ctx_model = AutoModel.from_pretrained(model_path)#.to(device)

    def forward(
            self,
            inputs,
    ):
        question_ids, question_segments, question_attn_mask, context_ids, ctx_segments, ctx_attn_mask = inputs
        _, sample_per_question, max_length = context_ids.shape

        q_vector = self.q_model(input_ids = question_ids,
                                attention_mask = question_attn_mask,
                                token_type_ids = question_segments,
                                ).pooler_output

        context_ids = context_ids.view(-1, max_length)
        ctx_segments = ctx_segments.view(-1, max_length)
        ctx_attn_mask = ctx_attn_mask.view(-1, max_length)

        ctx_vector = self.ctx_model(input_ids = context_ids,
                                  attention_mask = ctx_attn_mask,
                                  token_type_ids = ctx_segments,
                                  ).pooler_output

        ctx_vector = ctx_vector.view(-1, sample_per_question, self.config.hidden_size)
        return q_vector, ctx_vector # ctx_vector = (batch, sample_per_question, hidden_dim)
                                    # q_vector = (batch, hidden_dim)

    def cal_loss(self, q_vector, ctx_vector, postitive_idx_per_question): # ctx_vector (batch_size, sample_per_ques, hidden_dim)
        q_vector = torch.unsqueeze(q_vector, dim=1) # q_vecto (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        scores = self.dot_product(q_vector, ctx_vector) # shape = (batch, 1, sample_per_ques)
        scores = torch.squeeze(scores) # shape = (batch, sample_per_ques)
        logit = -torch.nn.LogSoftmax(dim = -1)(scores)
        loss = torch.mul(postitive_idx_per_question, logit).mean()
        return loss

    def my_function(self):
        print('hello my function')