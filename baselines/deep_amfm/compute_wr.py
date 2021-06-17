import pandas as pd
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

dataset_meta_info ={ 
    'dstc6': {
        'context':1, 
        'num_references':11, 
        'annotations': ['overall'], 
        'aggregation':np.mean},
    'dstc7' : {
        'context':1, 
        'num_references':1, 
        'annotations': ['relevance', 'informativeness', 'overall'], 
        'aggregation':np.mean},
    'dailydialog-gupta' : { 
        'context':-1, 
        'num_references':4, 
        'annotations': ['overall'], 
        'aggregation':lambda x: x[0]},
    'dailydialog-zhao' : { 
        'context':-1, 
        'num_references':1, 
        'annotations': ['content', 'grammar','overall','relevance'], 
        'aggregation':np.mean},
    'humod' : { 
        'context':1, 
        'num_references':3, 
        'annotations': ['language_usage', 'relevance'], 
        'aggregation':np.mean},
     'persona-usr' : { 
        'context':1, 
        'num_references':1, 
        'annotations': ['Understandable', 'Natural', 'Maintains Context', 'Engaging', 'Uses Knowledge', 'Overall'], 
        'aggregation':np.mean},
     'persona-zhao' : { 
        'context':1, 
        'num_references':1, 
        'annotations': ['overall'], 
        'aggregation':np.mean}, 
      'topical-usr' : { 
        'context':1, 
        'num_references':1, 
        'annotations': ['Understandable', 'Natural', 'Maintains Context', 'Engaging', 'Uses Knowledge', 'Overall'], 
        'aggregation':np.mean},       
        
}


def compute_fm_score(x, y):
    return max([x,y]) / min([x,y])


def normalize_df(dataset_name, df, ds_meta):
    dataset_meta = ds_meta[dataset_name]
    for annotation in dataset_meta['annotations']:
        df['annotations.' + annotation] = df['annotations.' + annotation].apply(dataset_meta['aggregation'])
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='up')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--am_model_path', type=str, default='embedding_models/persona_am/')
    parser.add_argument('--fm_model_path', type=str, default='language_models/persona_fm')
    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)

    bert_model = BertModel.from_pretrained(am_model_path).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(am_model_path)
    bert_model.eval()
    
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(fm_model_path)
    gpt2_model = GPT2LMHeadModel.from_pretrained(fm_model_path).to(device)
    gpt2_model.eval()

    with open('../../human_evaluation_data/{}_eval.json'.format(dataset)) as f:
        df = pd.json_normalize(json.load(f))
    df = normalize_df(dataset, df, dataset_meta_info)

    response_list = df.response.to_list()
    response_list = [item if item != '' else "no response" for item in response_list]

    reference_list = df.reference.to_list()
    annotations = ["annotations." + _ for _ in dataset_meta_info[dataset]["annotations"]]
    human_scores = {}
    for k in annotations:
        human_scores[k] = list(df[k])
    
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
    
    df['am_scores'] = am_scores

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
    
    df['fm_scores'] = fm_scores
    
    am_fm_scores = [np.mean([x, y]) for x, y in zip(am_scores, fm_scores)]
    for k, v in human_scores.items():
        pear, p = pearsonr(v, am_fm_scores)
        print("Pearson Correlation of AM-FM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(v, am_fm_scores)
        print("Spearman Correlation of AM-FM along {}: {} with p value: {}".format(k, spear, p))

    df['am_fm_scores'] = am_fm_scores
    df.to_csv(dataset + '_results.csv', index=None)
