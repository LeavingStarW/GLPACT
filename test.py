import pickle
import numpy as np
import os
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import util

from args import TestArgParser
from data_loader import CTDataLoader
from collections import defaultdict

from PIL import Image
from saver import ModelSaver
from tqdm import tqdm

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def test(args):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    model = model.to(args.device)
    model.eval()


    data_loader = CTDataLoader(args, phase = 'test', is_training = False)
    study2slices = defaultdict(list)
    study2probs = defaultdict(list)
    study2labels = {}


    



    
    
    #w
    

    with tqdm(total = len(data_loader.dataset), unit = 'windows') as progress_bar:
        for img, target_dict in data_loader:
            #w
            table = pd.read_csv(args.data_tab_path)
            ids = [item for item in target_dict['study_num']]
            tab = []
 
            for i in range(len(target_dict['study_num'])):
                data = table[table['NewPatientID'] == ids[i]].iloc[0, 1:8].astype(np.float32)
                tab.append(torch.tensor(data, dtype = torch.float32))
            tab = torch.stack(tab).squeeze(1)


            #w PE
            '''table_data = pd.read_csv(args.data_tab_path)
            ids = [int(item) for item in target_dict['study_num']]
            tab = []
            table_data = pd.read_csv(args.data_tab_path)
            for idx in ids:
                data = table_data[table_data['idx'] == idx].iloc[:, 4:].astype(np.float32)
                tab.append(torch.tensor(np.array(data), dtype = torch.float32))
            tab = torch.stack(tab).squeeze(1)'''

            #w
            with torch.no_grad():
                img = img.to(args.device)
                tab = tab.to(args.device)
                label = target_dict['is_abnormal'].to(args.device)


                #w
                output = model(img)
                cls_logits = output['out']
                cls_probs = F.sigmoid(cls_logits)



            max_probs = cls_probs.to('cpu').numpy()
            for study_num, slice_idx, prob in \
                    zip(target_dict['study_num'], target_dict['slice_idx'], list(max_probs)):
                #w slice_idx就是选择后的开始id，但是每张切片id都不一样
                study2slices[study_num].append(slice_idx)
                #w 一个ID的study可能会出现多次，所以可能会对应多个概率值，所以最后有一个取最大值的操作
                study2probs[study_num].append(prob.item())
                series = data_loader.get_series(study_num)
                if study_num not in study2labels:
                    study2labels[study_num] = int(series.is_positive)



    



    #w
    max_probs = []
    labels = []
    predictions = {}
    for study_num in tqdm(study2slices):
        #w 依据元组的第一个元素进行排序
        slice_list, prob_list = (list(t) for t in zip(*sorted(zip(study2slices[study_num], study2probs[study_num]),
                                                              key = lambda slice_and_prob:slice_and_prob[0]))) 
        #w 这是排序后的
        study2slices[study_num] = slice_list
        study2probs[study_num] = prob_list
        max_prob = max(prob_list)
        max_probs.append(max_prob)
        label = study2labels[study_num]
        labels.append(label)
        predictions[study_num] = {'label':label, 'pred':max_prob}

    with open('{}/{}.pickle'.format(args.results_preds_dir, args.name), 'wb') as fp:
        pickle.dump(predictions, fp)
        


    #w 指标计算
    max_probs, labels = np.array(max_probs), np.array(labels)
    metrics = {
        'test' + '_' + 'AUPRC':sk_metrics.average_precision_score(labels, max_probs),
        'test' + '_' + 'AUROC':sk_metrics.roc_auc_score(labels, max_probs),
    }
    with open(os.path.join(args.results_preds_dir, 'metrics.txt'), 'w') as metrics_fh:
        for k, v in metrics.items():
            metrics_fh.write('{}:{:.5f}\n'.format(k, v))






if __name__ == '__main__':
    #w
    util.set_spawn_enabled()
    parser = TestArgParser()
    args_ = parser.parse_args()
    #w
    test(args_)
