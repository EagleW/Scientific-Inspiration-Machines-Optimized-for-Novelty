from tqdm import tqdm
import json
from bert_score import score
from statistics import mean


dir_names = [
    '../models/T5/%s/', 
]
model_nameses = [
    ['t5_checkpoint'],
]
map_gt = {
    'forward': 'tail',
    'backward': 'head'
}

for dir_name,model_names in zip(dir_names, model_nameses):
    for model_name in model_names:
        metrics = {}
        forward_metrics = {}
        backward_metrics = {}

        path = dir_name % model_name
        print(path)

        count_f = 0
        count_b = 0
        f_template = 'eval_%s.json'
        for filename in ['forward', 'backward']:
            refs = []
            cands = []
            fname = path + f_template % filename
            predictions= json.load(open(fname, 'r'))
            
            for tmp in predictions:
                input_ = tmp['input']
                ground_truth = tmp['ground_truth'][0][map_gt[filename]]
                ttmp = tmp['topk_score_info']
                topk_score_info = json.loads(ttmp)
                for k in topk_score_info:
                    refs.append(ground_truth)
                    if k == "********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************":
                        cands.append(k[:128])
                    else:
                        cands.append(k)
                        
            (P, R, F1), hash = score(cands, refs, lang='en-sci', model_type='allenai/scibert_scivocab_uncased', batch_size=128, verbose= True, return_hash=True) 
            P = P.tolist()
            R = R.tolist()
            F1 = F1.tolist()
            
            max_bert_ps = []
            max_bert_rs = []
            max_bert_f1s = []

            avg_bert_ps = []
            avg_bert_rs = []
            avg_bert_f1s = []

            index = 0
            for i, tmp in enumerate(predictions):
                avg_bert_p = 0
                avg_bert_r = 0
                avg_bert_f1 = 0
                max_bert_p = 0
                max_bert_r = 0
                max_bert_f1 = 0
                sum_coef = 0
                ttmp = tmp['topk_score_info']
                topk_score_info = json.loads(ttmp)
                for j in range(len(topk_score_info)):
                    coef = 1/(j + 1)
                    sum_coef += coef
                    max_bert_p = max(P[index], max_bert_p)
                    max_bert_r = max(R[index], max_bert_r)
                    max_bert_f1 = max(F1[index], max_bert_f1)

                    avg_bert_p += P[index] * coef
                    avg_bert_r += R[index] * coef
                    avg_bert_f1 += F1[index] * coef
                    index += 1
                avg_bert_p /= sum_coef
                avg_bert_r /= sum_coef
                avg_bert_f1 /= sum_coef

                tmp['avg_bert_P'] = avg_bert_p
                tmp['avg_bert_R'] = avg_bert_r
                tmp['avg_bert_F1'] = avg_bert_f1

                tmp['max_bert_p'] = max_bert_p
                tmp['max_bert_r'] = max_bert_r
                tmp['max_bert_f1'] = max_bert_f1


                avg_bert_ps.append(avg_bert_p)
                avg_bert_rs.append(avg_bert_r)
                avg_bert_f1s.append(avg_bert_f1)

                max_bert_ps.append(max_bert_p)
                max_bert_rs.append(max_bert_r)
                max_bert_f1s.append(max_bert_f1)
            if filename == 'forward':
                count_f = len(refs)
                forward_metrics['avg_bertscore_P'] = mean(avg_bert_ps)
                forward_metrics['avg_bertscore_R'] = mean(avg_bert_rs)
                forward_metrics['avg_bertscore_F1'] = mean(avg_bert_f1s)

                forward_metrics['max_bertscore_P'] = mean(max_bert_ps)
                forward_metrics['max_bertscore_R'] = mean(max_bert_rs)
                forward_metrics['max_bertscore_F1'] = mean(max_bert_f1s)


                with open(path + 'eval_scibert_forward.json', 'w', encoding='utf-8') as writer:
                    writer.write(json.dumps(predictions, ensure_ascii=False, indent=4))
            else:
                count_b = len(refs)
                backward_metrics['avg_bertscore_P'] = mean(avg_bert_ps)
                backward_metrics['avg_bertscore_R'] = mean(avg_bert_rs)
                backward_metrics['avg_bertscore_F1'] = mean(avg_bert_f1s)

                backward_metrics['max_bertscore_P'] = mean(max_bert_ps)
                backward_metrics['max_bertscore_R'] = mean(max_bert_rs)
                backward_metrics['max_bertscore_F1'] = mean(max_bert_f1s)

                with open(path + 'eval_scibert_backward.json', 'w', encoding='utf-8') as writer:
                    writer.write(json.dumps(predictions, ensure_ascii=False, indent=4))
        print()
        metrics = {k: round((forward_metrics[k] * count_f + backward_metrics[k] * count_b) / (count_b + count_f), 4) for k in forward_metrics}
        with open(path + 'scibert_metrics.json', 'w', encoding='utf-8') as writer:
            writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
            writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
            writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

