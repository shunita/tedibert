import torch


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def break_sentence_batch(samples):
    indexes = []
    all_sentences = []
    index = 0
    max_len = 0
    for sample in samples:
        sample_as_list = sample.split('<BREAK>')
        # sample is a list of sentences
        indexes.append((index, index + len(sample_as_list)))
        index += len(sample_as_list)
        all_sentences.extend(sample_as_list)
        if max_len < len(sample_as_list):
            max_len = len(sample_as_list)
    return indexes, all_sentences, max_len