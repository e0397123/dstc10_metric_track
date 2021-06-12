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
        response_list.append(item['response'])
        reference_list.append(item['context'].split('\n')[-1])
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
        for r in tqdm(reference_list):
            inputs = {k:v.to(device) for k, v in bert_tokenizer(r, return_tensors="pt").items()}
            outputs = bert_model(**inputs, return_dict=True)
            pooler_output = outputs.pooler_output
            pooler_output.cpu().numpy()
            reference_embedding_list.append(pooler_output.cpu().numpy())
    am_scores = []
    for idx, (x, y) in enumerate(zip(response_embedding_list, reference_embedding_list)):
        single_am_score = cosine_similarity(x, y)[0][0]
        am_scores.append(single_am_score)

    cutoff = np.quantile(am_scores, 0.05)
    modified_rating = np.array([cutoff if t < cutoff else t for t in am_scores])
    normed_am_scores = (modified_rating - cutoff) / np.abs(cutoff)
    for k, v in human_scores.items():
        pear, p = pearsonr(v, normed_am_scores)
        print("Pearson Correlation of AM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(v, normed_am_scores)
        print("Spearman Correlation of AM along {}: {} with p value: {}".format(k, spear, p))

    fm_scores = []
    with torch.no_grad():
        for prev, cur in tqdm(zip(reference_list, response_list)):
            joint_enc = gpt2_tokenizer.encode(str(prev)+' '+str(cur)) + [50256]
            q = gpt2_tokenizer.encode(str(prev)) + [50256]
            batch_joint = torch.tensor([joint_enc]).to(device)
            batch_q = torch.tensor([q]).to(device)
            loss_joint = gpt2_model(batch_joint, labels=batch_joint)[0]
            loss_q =  gpt2_model(batch_q, labels=batch_q)[0]
            p_joint = -loss_joint * (len(joint_enc) -1)
            p_q = -loss_q * (len(q) -1)
            score = (p_joint - (p_q)) / ((len(joint_enc) -1) - (len(q) -1))
            fm_scores.append(score.item())
    cutoff = np.quantile(fm_scores, 0.05)
    modified_rating = np.array([cutoff if t < cutoff else t for t in fm_scores])
    normed_fm_scores = (modified_rating - cutoff) / np.abs(cutoff)
    for k, v in human_scores.items():
        pear, p = pearsonr(v, normed_fm_scores)
        print("Pearson Correlation of FM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(v, normed_fm_scores)
        print("Spearman Correlation of FM along {}: {} with p value: {}".format(k, spear, p))

    am_fm_scores = [np.mean([x, y]) for x, y in zip(normed_am_scores, normed_fm_scores)]
    for k, v in human_scores.items():
        pear, p = pearsonr(v, am_fm_scores)
        print("Pearson Correlation of AM-FM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(v, am_fm_scores)
        print("Spearman Correlation of AM-FM along {}: {} with p value: {}".format(k, spear, p))
