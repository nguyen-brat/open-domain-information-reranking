import torch
from torchpreprocess import DataloaderReranking
from torchmodel import Biencoder
from torch.utils.data import DataLoader
import tqdm


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_per_ques=12
    context_piece_max_length = 128
    batch_size = 8
    epochs = 10
    dataloader = DataloaderReranking(
        pth_raw_ctx = 'raw_context.json',
        pth_sample_ctx = 'raw_sample_data.json',
        context_piece_max_length=context_piece_max_length,
        batch_size=batch_size,
        sample_per_ques=sample_per_ques,
        num_hard_negative_sample=1,
        num_gold_sample=1,
    )
    model = Biencoder()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        print(f'Epochs {epoch+1}/{epochs} ', end='')
        running_loss = 0
        for data in tqdm(train_dataloader):
            question_ids, question_token_type_ids, question_attention_mask, context_ids, context_token_type_ids, context_attention_mask, positive_id = data

            inputs = (question_ids, question_token_type_ids, question_attention_mask, context_ids, context_token_type_ids, context_attention_mask)
            model.train(True)
            optimizer.zero_grad()
            # Make predictions for this batch
            q_vector, ctx_vector = model(inputs)
            # Compute the loss and its gradients
            loss = model.cal_loss(q_vector, ctx_vector, positive_id)
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            model.eval()

            question_ids.detach().cpu()
            question_token_type_ids.detach().cpu()
            question_attention_mask.detach().cpu()
            context_ids.detach().cpu()
            context_token_type_ids.detach().cpu()
            context_attention_mask.detach().cpu()
            positive_id.detach().cpu()
            running_loss = running_loss + loss.item()

        print(f'The final loss is {running_loss/len(train_dataloader)}')