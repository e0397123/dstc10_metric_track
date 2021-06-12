import json
import torch
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel

torch.manual_seed(2000)
np.random.seed(2000)

def compute_fm_score(x, y):
    return max([x,y]) / min([x,y])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='up')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--am_model_path', type=str, default='embedding_models/persona_am/')
    parser.add_argument('--fm_model_path', type=str, default='language_models/persona_fm')
    parser.add_argument('--criterion', nargs='+')
    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)

    bert_model = BertModel.from_pretrained(am_model_path).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(am_model_path)
    bert_model.eval()
    
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(fm_model_path)
    gpt2_model = GPT2LMHeadModel.from_pretrained(fm_model_path).to(device)
    gpt2_model.eval()

    full_data = json.load(open('human_evaluation_data/{}_eval.json'.format(dataset), 'r'))
    response_list =[]
    reference_list =[]
    human_scores = {}
    for c in criterion:
        human_scores[c] = []
    for item in full_data:
        if len(item['reference']) >= 1:
            response_list.append(item['response'])
            reference_list.append(item['reference'])
            for c in criterion:
                human_scores[c].append(np.mean(item['annotations'][c]))
    response_embedding_list = []
    with torch.no_grad():
        for r in tqdm(response_list):
            inputs = {k:v.to(device) for k, v in bert_tokenizer(r, return_tensors="pt").items()}
            outputs = bert_model(**inputs, return_dict=True)
            pooler_output = outputs.pooler_output
            response_embedding_list.append(pooler_output.cpu().numpy())

    reference_embedding_list = []
    with torch.no_grad():
        for refs in tqdm(reference_list):
            temp = []
            for r in refs:
                inputs = {k:v.to(device) for k, v in bert_tokenizer(r, return_tensors="pt").items()}
                outputs = bert_model(**inputs, return_dict=True)
                pooler_output = outputs.pooler_output
                temp.append(pooler_output.cpu().numpy())
                reference_embedding_list.append(temp)
    am_scores = []
    for idx, (x, y) in enumerate(zip(response_embedding_list, reference_embedding_list)):
        single_am_scores = []
        for r in y:
            single_am_scores.append(cosine_similarity(x, r)[0][0])
        max_am_score = np.amax(single_am_scores)
        am_scores.append(max_am_score)
    # am_scores = np.diagonal(cosine_similarity(np.stack(reference_embedding_list), np.stack(response_embedding_list)))
    for k, v in human_scores.items():
        pear, p = pearsonr(v, am_scores)
        print("Pearson Correlation of AM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(v, am_scores)
        print("Spearman Correlation of AM along {}: {} with p value: {}".format(k, spear, p))

    response_fm_score_list = []
    with torch.no_grad():
        for r in tqdm(response_list):
            r = gpt2_tokenizer.encode(str(r))  + [50256]
            batch = torch.tensor([r]).to(device)
            # average -logp
            loss = gpt2_model(batch, labels=batch)[0]
            response_fm_score_list.append(-1*loss.item())

    reference_fm_score_list = []
    with torch.no_grad():
        for refs in tqdm(reference_list):
            temp = []
            for r in refs:
                r = gpt2_tokenizer.encode(str(r))  + [50256]
                batch = torch.tensor([r]).to(device)
                # average -logp
                loss = gpt2_model(batch, labels=batch)[0]
                temp.append(-1*loss.item())
                reference_fm_score_list.append(temp)

    fm_scores = []
    for idx, (x, y) in enumerate(zip(response_fm_score_list, reference_fm_score_list)):
        single_fm_scores = []
        for r in y:
            single_fm_scores.append(compute_fm_score(x, r))
        fm_scores.append(np.amax(single_fm_scores))
    # fm_scores = [compute_fm_score(x, y) for x, y in zip(response_fm_score_list, reference_fm_score_list)]
    for k, v in human_scores.items():
        pear, p = pearsonr(v, fm_scores)
        print("Pearson Correlation of FM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(v, fm_scores)
        print("Spearman Correlation of FM along {}: {} with p value: {}".format(k, spear, p))

    am_fm_scores = [np.mean([x, y]) for x, y in zip(am_scores, fm_scores)]
    for k, v in human_scores.items():
        pear, p = pearsonr(v, am_fm_scores)
        print("Pearson Correlation of AM-FM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(v, am_fm_scores)
        print("Spearman Correlation of AM-FM along {}: {} with p value: {}".format(k, spear, p))

