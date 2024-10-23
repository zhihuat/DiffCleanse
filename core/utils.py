import torch

def add_target(batch, train_batch_size, token_id, compare_token_id):
    insert_id = torch.cat([token_id, compare_token_id], dim=0).repeat_interleave(train_batch_size//2, dim=0).unsqueeze(1)
    id_0 = torch.cat((batch["input_ids"][:, :1], insert_id, batch["input_ids"][:, 1:]), dim=1)[:, :77]
    batch["input_ids"] = id_0

    return batch


def normalize(sa):
    # sa = sa.reshape(-1, 64 * 64)
    min_vals = sa.min(1,keepdim=True)[0]
    max_vals = sa.max(1, keepdim=True)[0]
    
    sa = (sa - min_vals) / (max_vals - min_vals)
    return sa