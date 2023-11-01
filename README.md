# reranking-opendomain-answer

# Pretrained used

```python
from sentence_transformers.cross_encoder import CrossEncoder
import torch

model = CrossEncoder("nguyen-brat/pdt-reranking-model_v2", max_length=512)

relevant_score = model.predict(['An đi học khi nào', 'Bé An đi học vào buổi chiều'], apply_softmax=True) apply_softmax=True)
```

# Data detail
One batch of train data include hard negative training sample (it in top 20 relative answer retrieval from bm-25 but not the answer),


gold negative sample (it is the answer of another the question in the same batch) and positive sample. Training sample is about 168 sample so the model may not learn good feature.

```
batch 1:
    sample 1:
        question: "Cần bao nhiêu tín chỉ để được tốt nghiệp ?"
        context: [
            "tùy vào ngành những cần khoảng 120 tín chỉ để được tốt nghiệp đối với chương trình cử nhân", # answer
            "muốn được tốt nghiệp thì phải hoàn thành đủ chứng chỉ, có chứng chỉ tiếng anh và hoàn thành đủ ngành công tác xã hội", # gold sample, because it is the answer in sample 4
            "sinh viên muôn tốt nghiệp thì cần có chứng chỉ tiếng anh",
            ...
        ]
    .
    .
    .
    sample 4:
    question: "Muốn tốt nghiệp thì cần những điều kiện gì ?"
    context: [
        "muốn được tốt nghiệp thì phải hoàn thành đủ chứng chỉ, có chứng chỉ tiếng anh và hoàn thành đủ ngành công tác xã hội", # answer
        "học bổng sẽ được trao cho những cá nhân có thành tích vượt trội trong khóa",
        "muốn tốt nghiệp phải có chứng chỉ MOS", # hard negative sample
        ...
    ]

batch 2:
    sample 1:
        question: "Thời gian đào tạo bật đại học là bao lâu ?"
        context: [
            "Thời gian đào tạo là 4 năm", # answer
            "Sinh viên có thể học vượt",
            "Sinh viên phải có trình độ ilets 6.0 trước khi ra trường",
            ...
        ]
    .
    .
    .
    sample 4:
    question: "Tôi muốn học hai bằng thì phải làm sao"
    context: [
        "phải hoàn thành trước năm hai mới có thể đăng kí đào tạo 2 bằng", # answer
        "học bổng sẽ được trao cho những cá nhân có thành tích vượt trội trong khóa",
        "muốn tốt nghiệp phải có chứng chỉ MOS",
        ...
    ]
```