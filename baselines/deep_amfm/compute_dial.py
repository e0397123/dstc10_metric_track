import json
import torch
import numpy as np
import argparse
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel

torch.manual_seed(2000)
np.random.seed(2000)

def compute_fm_score(x, y):
    return max([x,y]) / min([x,y])

dataset_meta_info ={
    'fed-dial': {
        'annotations': ['Coherent', 'Error recovery', 'Consistent', 'Diverse', 'Depth', 'Likeable', 'Understanding', 'Flexible', 'Informative', 'Inquisitive', 'Overall'],
        'aggregation':np.mean},
    'persona-see': {
        'annotations': ['enjoy', 'interest', 'listen', 'turing', 'avoid_rep', 'make_sense', 'fluency', 'inquisitive', 'persona_guess'],
        'aggregation':np.mean},
}

def normalize_df(dataset_name, df, ds_meta):
    dataset_meta = ds_meta[dataset_name]
    for annotation in dataset_meta['annotations']:
        df['annotations.' + annotation] = df['annotations.' + annotation].apply(dataset_meta['aggregation'])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='persona-see')
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

    with open('../../human_evaluation_data/{}_eval.json'.format(dataset)) as f:
        df = pd.json_normalize(json.load(f))
    human_scores = {}
    annotations = ["annotations." + _ for _ in dataset_meta_info[dataset]["annotations"]]
    for k in annotations:
        human_scores[k] = list(df[k])
    df = normalize_df(dataset, df, dataset_meta_info)

    dialog_list = df.dialog.to_list()
    full_human_model_pairs = []
    for whole_dialog in dialog_list:
        human_model_pairs = []
        for idx, utt in enumerate(whole_dialog):
            if utt['speaker'] == 'model' and idx != 0:
                prev_utt = whole_dialog[idx-1]
                human_model_pairs.append((prev_utt['text'], utt['text']))
        full_human_model_pairs.append(human_model_pairs)
    
    # to handle the missing annotations for error recovery category
    new_human_scores = {}
    for k, v in human_scores.items():
        new_human_scores[k] = []
        for item in v:
            if 'Error recovery' in k:
                if len(item) == 0:
                    new_human_scores[k].append((False, 0))
                else:
                    new_human_scores[k].append((True, np.mean(item)))
            else:
                new_human_scores[k].append((True, np.mean(item)))
            
    am_scores_dialog_level = []
    with torch.no_grad():
        for dialog in tqdm(full_human_model_pairs):
            am_scores_turn_level = []
            for prev, cur in dialog:
                prev_inputs = {k:v.to(device) for k, v in bert_tokenizer(prev, return_tensors="pt").items()}
                cur_inputs = {k:v.to(device) for k, v in bert_tokenizer(cur, return_tensors="pt").items()}
                prev_outputs = bert_model(**prev_inputs, return_dict=True)
                cur_outputs = bert_model(**cur_inputs, return_dict=True)
                prev_pooler_output = prev_outputs.pooler_output.cpu().numpy()
                cur_pooler_output = cur_outputs.pooler_output.cpu().numpy()
                am_scores_turn_level.append(cosine_similarity(prev_pooler_output, cur_pooler_output)[0][0])
            am_scores_dialog_level.append(np.mean(am_scores_turn_level))
    
    cutoff = np.quantile(am_scores_dialog_level, 0.05)
    modified_rating = np.array([cutoff if t < cutoff else t for t in am_scores_dialog_level])
    normed_am_scores_dialog_level = (modified_rating - cutoff) / np.abs(cutoff)
    for k, v in new_human_scores.items():
        assert len(v) == len(normed_am_scores_dialog_level)
        s_1 = []
        s_2 = []
        for idx, x in enumerate(v):
            if x[0]:
                s_1.append(x[1])
                s_2.append(normed_am_scores_dialog_level[idx])
        pear, p = pearsonr(s_1, s_2)
        print("Pearson Correlation of AM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(s_1, s_2)
        print("Spearman Correlation of AM along {}: {} with p value: {}".format(k, spear, p))
    df['am_scores'] = normed_am_scores_dialog_level

    fm_scores_dialog_level = []
    with torch.no_grad():
        for dialog in tqdm(full_human_model_pairs):
            fm_scores_turn_level = []
            for prev, cur in dialog:
                joint_enc = gpt2_tokenizer.encode(str(prev)+' '+str(cur)) + [50256]
                q = gpt2_tokenizer.encode(str(prev)) + [50256]
                batch_joint = torch.tensor([joint_enc]).to(device)
                batch_q = torch.tensor([q]).to(device)
                loss_joint = gpt2_model(batch_joint, labels=batch_joint)[0]
                loss_q =  gpt2_model(batch_q, labels=batch_q)[0]
                p_joint = -loss_joint * (len(joint_enc) -1)
                p_q = -loss_q * (len(q) -1)
                score = (p_joint - (p_q)) / ((len(joint_enc) -1) - (len(q) -1))
                fm_scores_turn_level.append(score.item())
            fm_scores_dialog_level.append(np.mean(fm_scores_turn_level))
    cutoff = np.quantile(fm_scores_dialog_level, 0.05)
    modified_rating = np.array([cutoff if t < cutoff else t for t in fm_scores_dialog_level])
    normed_fm_scores_dialog_level = (modified_rating - cutoff) / np.abs(cutoff)
    for k, v in new_human_scores.items():
        s_1 = []
        s_2 = []
        for idx, x in enumerate(v):
            if x[0]:
                s_1.append(x[1])
                s_2.append(normed_fm_scores_dialog_level[idx])
        pear, p = pearsonr(s_1, s_2)
        print("Pearson Correlation of FM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(s_1, s_2)
        print("Spearman Correlation of FM along {}: {} with p value: {}".format(k, spear, p))
    df['fm_scores'] = normed_fm_scores_dialog_level

    am_fm_scores = [np.mean([-x, y]) for x, y in zip(normed_am_scores_dialog_level, normed_fm_scores_dialog_level)]
    for k, v in new_human_scores.items():
        s_1 = []
        s_2 = []
        for idx, x in enumerate(v):
            if x[0]:
                s_1.append(x[1])
                s_2.append(am_fm_scores[idx])
        pear, p = pearsonr(s_1, s_2)
        print("Pearson Correlation of AM-FM along {}: {} with p value: {}".format(k, pear, p))
        spear, p = spearmanr(s_1, s_2)
        print("Spearman Correlation of AM-FM along {}: {} with p value: {}".format(k, spear, p))
    df['am_fm_scores'] = am_fm_scores
    df.to_csv(dataset + '_results.csv', index=None)
