import os
from sentence_transformers.cross_encoder import CrossEncoder
from trainer_rerank import trainer
from reranking_engine.Dataloader import DataloaderReranking
from torch.utils.data import DataLoader
from typing import List
import numpy as np

class xlm_roberta_reranking:
    def __init__(
            self,
            batch_size=4,
            sample_per_ques=4,
            num_hard_negative_sample=1,
            num_gold_sample=1,
            MODEL:str='xlm-roberta-base',
    ):
        if os.path.exists('reranking_model'):
            self.model = CrossEncoder('reranking_model')
        else:
            train_object = DataloaderReranking(
                pth_raw_ctx = 'raw_context.json',
                pth_sample_ctx = 'raw_sample_data.json',
                batch_size=batch_size,
                sample_per_ques=sample_per_ques,
                num_hard_negative_sample=num_hard_negative_sample,
                num_gold_sample=num_gold_sample,
            )
            train_sample = train_object()
            train_dataloader = DataLoader(train_sample, batch_size=4, shuffle=False)
            train = trainer(
                train_dataloader,
                MODEL,
            )
            train(epochs=10)
            self.model = train.model
    
    def __call__(self, question:str, answers:List):
        reranking_score = []
        for answer in answers:
            pair = [question, answer]
            result = self.softmax(self.model.predict(pair))[1]
            reranking_score.append(result)
        sort_index = np.argsort(np.array(reranking_score))
        reranking_answer = list(np.array(answers)[sort_index])
        reranking_answer.reverse()
        return reranking_answer
    
    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

if __name__ == "__main__":
    reranking = xlm_roberta_reranking()
    print(reranking(
        "có cần chứng chỉ tiếng anh để  tốt nghiệp không ?",
        [
            "bắt buộc phải có để hoàn thành việc học",
            "sinh viên phải có trình độ tiếng anh ít nhất 6.0 ilets hoặc tương đương khi tốt nghiệp",
            "chứng chỉ tiếng anh ilets được công nhận quốc tế",
            "không bắt buộc",
            "tiếng anh tốt nghiếp chứng chỉ có cần",
            "tiếng anh khác tiếng việt",
            "mặt trời mọc ở đằng tây thiên hạ ngạc nhiên chuyện là này",
            "không bắt buộc phải có",
        ]
    ))