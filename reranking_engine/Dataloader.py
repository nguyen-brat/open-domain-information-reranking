import torch
import numpy as np
from typing import Any, Tuple, List
import re
from rank_bm25 import BM25Okapi
import json
import random
from underthesea import word_tokenize
from sentence_transformers.readers import InputExample

class DataloaderReranking():
    def __init__(self,
            pth_raw_ctx:str, # path to raw context in json file
            pth_sample_ctx:str, # path to training sample in json file
            segmentation:bool=True,
    ):
        with open(pth_raw_ctx, 'r') as f:
            self.raw_context = json.load(f)["context"]
        with open(pth_sample_ctx, 'r') as f:
            self.training_sample = json.load(f)
        self.segmentation = segmentation
        self.fit_context()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(
            self,
            batch_size=4,
            sample_per_ques=4,
            num_hard_negative_sample=1,
            num_gold_sample=1,
    ):
        batch_question_and_context = self.generate_training_sample(
            batch_size=batch_size,
            sample_per_ques=sample_per_ques,
            num_hard_negative_sample=num_hard_negative_sample,
            num_gold_sample=num_gold_sample,
        )
        ''' a batch_dataset look like
        {
            "question":["question 1", "question 2",..., "question n"],
            "context"'[[id_00, id_01, id_0n], [id_10, id_11, id_1n], ..., [id_n0, id_n1, id_nn]],
            "positive_id": [[0, 0, 1, 0,.., 0], [1, 0, 0, ..., 1],..., [0, 0, 0, ..., 0]] (list of one hot vector)
        }
        .......
        '''
        training_sample = []
        for batch in batch_question_and_context:
            for question_id in range(len(batch["question"])):
                for sample_id, label in zip(batch["context"][question_id], batch["positive_id"][question_id]):
                    context = self.clean_context[sample_id]
                    training_sample.append(InputExample(texts=[batch["question"][question_id],
                                                               context],
                                                               label=label))
        return training_sample


    @staticmethod
    def _preprocess(text:str)->str:
        text = text.lower()
        reg_pattern = '[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ\s]'
        output = re.sub(reg_pattern, '', text)
        return output

    def preprocess(self, texts:List)->List:
        return [self._preprocess(text) for text in texts]

    @staticmethod
    def sen_segment(texts: List)->List:
        '''
        tokenize text take a list of raw text
        and return a list a tokenize text
        '''
        return [word_tokenize(text, format='text') for text in texts]

    @staticmethod
    def context_cut(text, max_length, stride):
        text = text.split(' ')
        text = [text[i:i+max_length] for i in range(0, len(text), stride)]
        return ' '.join(text)

    @staticmethod
    def generate_random_idx(upper:int, lower:int=0):
        shuffle = list(range(lower, upper))
        random.shuffle(shuffle)
        shuffle = np.array(shuffle)
        return shuffle

    def fit_context(self):
        self.clean_context = self.preprocess(self.raw_context)
        if self.segmentation:
            self.clean_context = self.sen_segment(self.clean_context)
        self.bm25 = BM25Okapi([text.split() for text in self.clean_context])

    def remove_answer_not_match_bm25_retrieval(self, question_list, positive_context_idx_list):
        clean_question_list = []
        clean_positive_text_idx_list = []
        for question, contexts_idx in zip(question_list, positive_context_idx_list):
            relevant_ctx_ids, _ = self.retrieval(question)
            relevant_answer_idx = [answer_idx for answer_idx in contexts_idx if answer_idx in relevant_ctx_ids[:100]]
            if len(relevant_answer_idx) > 0:
                clean_question_list.append(question)
                clean_positive_text_idx_list.append(relevant_answer_idx)
        return clean_question_list, clean_positive_text_idx_list

    def retrieval(self,
            text:str,
            top_k:int=100,
            return_context:bool=False,
    )->Tuple[List, List]:
        '''
        take a raw input text as query and
        return index of accesding order of
        relevant of context
        '''
        text = self._preprocess(text)
        if self.segmentation:
            text = word_tokenize(text, format='text')
        doc_scores = np.array(self.bm25.get_scores(text.split()))
        sort_idx = np.flip(np.argsort(doc_scores))
        if return_context:
            return [self.raw_context[idx] for idx in sort_idx[:top_k]]
        return sort_idx, doc_scores

    def generate_training_sample(
            self,
            batch_size:int=32,
            sample_per_ques:int=128,
            num_hard_negative_sample:int=2,
            num_gold_sample:int=2,
    ):
        '''
        create sample for training
        one sample inclue a question, positive sample, hard negative sample
        gold sample and normal negative sample
        '''
        # generate shuffle index
        shuffle = self.generate_random_idx(len(self.training_sample["question"]))
        # suffle question and context
        question_list = list(np.array(self.training_sample["question"])[shuffle])
        positive_context_list = list(np.array(self.training_sample["context"])[shuffle])
        # turn the document to index of raw_context
        positive_context_list_idx = [[self.raw_context.index(context) for context in positive_contexts] for positive_contexts in positive_context_list]
        # remove sample have all answer not in top 100 relavant documant by bm25
        question_list, positive_context_list_idx = self.remove_answer_not_match_bm25_retrieval(question_list, positive_context_list_idx)
        # cut list into batch of list
        batch_questions = [question_list[i:i+batch_size] for i in range(0, len(question_list)//batch_size*batch_size, batch_size)]
        batch_positive_contexts_idx = [positive_context_list_idx[i:i+batch_size] for i in range(0, len(positive_context_list_idx)//batch_size*batch_size, batch_size)]
        dataset = []
        for batch_question, batch_positive_context_idx in zip(batch_questions, batch_positive_contexts_idx):
            #batch_positive_context_idx = [[self.raw_context.index(context) for context in positive_contexts] for positive_contexts in batch_positive_context]
            batch_dataset = {
                "question":[],
                "context":[],
                "positive_id":[]
            }
            for question, positive_context_idx in zip(batch_question, batch_positive_context_idx):
                context_id_to_train_from_question = []
                # add positive context
                relevant_ctx_ids, _ = self.retrieval(question)
                context_id_to_train_from_question += positive_context_idx
                num_positive_sample = len(positive_context_idx)

                # add hard negative sample
                hard_negative_idx = [index for index in relevant_ctx_ids[:num_hard_negative_sample+num_positive_sample] if index not in positive_context_idx][:num_hard_negative_sample]
                context_id_to_train_from_question += hard_negative_idx

                # add gold sample
                batch_len = batch_size if len(batch_question) == batch_size else len(batch_question)
                shuffle = self.generate_random_idx(batch_len)
                random_batch_sample_idx = shuffle[:num_gold_sample+num_hard_negative_sample+num_positive_sample]
                gold_sample_idx = [batch_positive_context_idx[batch][0] for batch in random_batch_sample_idx if batch_positive_context_idx[batch][0] not in context_id_to_train_from_question][:num_gold_sample]
                context_id_to_train_from_question += gold_sample_idx

                # add easy negative sample
                num_easy_negative_sample = sample_per_ques - num_positive_sample - len(hard_negative_idx) - len(gold_sample_idx)
                shuffle = self.generate_random_idx(len(relevant_ctx_ids[100:]))
                context_id_to_train_from_question += list(relevant_ctx_ids[100:][shuffle][:num_easy_negative_sample])

                positive_id = np.zeros(shape = (sample_per_ques), dtype=np.int32)
                positive_id[:num_positive_sample] = 1
                #shuffle index
                shuffle = self.generate_random_idx(sample_per_ques)
                positive_id = positive_id[shuffle]
                context_id_to_train_from_question = list(np.array(context_id_to_train_from_question)[shuffle])
                #
                batch_dataset["question"].append(question)
                batch_dataset["context"].append(context_id_to_train_from_question)
                batch_dataset["positive_id"].append(torch.tensor(positive_id))
                ''' a batch_dataset look like
                {
                    "question":["question 1", "question 2",..., "question n"],
                    "context"'[[id_00, id_01, id_0n], [id_10, id_11, id_1n], ..., [id_n0, id_n1, id_nn]],
                    "positive_id": [[0, 0, 1, 0,.., 0], [1, 0, 0, ..., 1],..., [0, 0, 0, ..., 0]] (list of one hot vector)
                }
                '''
            dataset.append(batch_dataset)
        return dataset

if __name__ == '__main__':
    pass