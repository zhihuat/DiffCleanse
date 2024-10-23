# Loss functions to compare the latents
import torch
from torch import nn
from core.utils import normalize

def get_positive_sum(interest_token_embeds, temperature):
    interest_token_embeds = interest_token_embeds.view(-1, 64 * 64)
    num_samples = interest_token_embeds.shape[0]
    # interest_token_embeds = normalize(interest_token_embeds)
    # interest_token_embeds[interest_token_embeds<0.5] = 0
    
    # Normalize token predictions
    interest_token_embeds = interest_token_embeds / (interest_token_embeds.norm(dim=1)[:, None] + 1e-20)
    # Calculate similarity values
    token_sim = torch.mm(interest_token_embeds, interest_token_embeds.T)
    token_sim = torch.exp(token_sim / temperature)
    
    # expanded_a = interest_token_embeds.unsqueeze(1)  # Shape: (-1, 1, 64 * 64)
    # expanded_b = interest_token_embeds.unsqueeze(0)  # Shape: (-1, 4, 64 * 64)
    # token_sim = torch.norm(expanded_a - expanded_b, p=2, dim=2)

    # Mask for positive samples
    if num_samples == 0:
        print("ERROR")

    # Calculate for different latent with same directions - upper triangle to not get repeating similarities
    mask = (torch.ones((num_samples, num_samples)).triu() * (1 - torch.eye(num_samples))).to(interest_token_embeds.device).bool()
    pos_sum = token_sim.masked_select(mask).view(mask.sum(), -1).sum()
    pos_count = mask.sum()

    return pos_sum, pos_count

def get_negative_sum(interest_token_embeds, negative_token_embeds, temperature):
    interest_token_embeds = interest_token_embeds.view(-1, 64 * 64)
    negative_token_embeds = negative_token_embeds.reshape(-1, 64 * 64)
    
    # interest_token_embeds = normalize(interest_token_embeds)
    # negative_prompt_embeds = normalize(negative_prompt_embeds)
    
    # interest_token_embeds[interest_token_embeds<0.5] = 0
    # negative_prompt_embeds[negative_prompt_embeds<0.5] = 0
    
    # Normalize the token_predictions
    interest_token_embeds = interest_token_embeds / (interest_token_embeds.norm(dim=1)[:, None] + 1e-20)
    negative_token_embeds = negative_token_embeds / (negative_token_embeds.norm(dim=1)[:, None] + 1e-20)
                
    # Calculate the similarity values
    token_sim = torch.mm(interest_token_embeds, negative_token_embeds.T)
    token_sim = torch.exp(token_sim / temperature)
    
    # expanded_a = interest_token_embeds.unsqueeze(1)  # Shape: (-1, 1, 64 * 64)
    # expanded_b = negative_prompt_embeds.unsqueeze(0)  # Shape: (-1, 4, 64 * 64)
    # token_sim = torch.norm(expanded_a - expanded_b, p=2, dim=2)
    
                
    # Mask for negative samples 
    # Calculate for same latent with different directions - with diagonal
    mask = torch.eye(interest_token_embeds.shape[0]).to(interest_token_embeds.device).bool()
    neg_sum = token_sim.masked_select(mask).view(mask.sum(), -1).sum()
    neg_count = mask.sum()

    return neg_sum, neg_count