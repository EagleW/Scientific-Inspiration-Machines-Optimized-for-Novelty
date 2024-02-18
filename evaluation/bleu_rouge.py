import json
from bleu.bleu import Bleu
from rouge.rouge import Rouge
from tqdm import tqdm
from statistics import mean
dir_names = [
    '../models/T5/%s/', 
]
model_nameses = [
    ['t5_checkpoint'],
]
b_scorer = Bleu(4)
r_scorer = Rouge()
method_list = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L"]
idx2method = {}
for ids, m in enumerate(method_list):
    idx2method[m] = ids
map_gt = {
    'forward': 'tail',
    'backward': 'head'
}

f_template = 'eval_%s.json'
for dir_name,model_names in zip(dir_names, model_nameses):
    for model_name in model_names:
        path = dir_name % model_name
        print(path)
        metrics = {}
        forward_metrics = {}
        backward_metrics = {}
        count_f = 0
        count_b = 0

        idx = 0
        for filename in ['forward', 'backward']:
            refs = {}
            cands = {}
            fname = path + f_template % filename
            predictions= json.load(open(fname, 'r'))

            for tmp in tqdm(predictions):
                ground_truth = tmp['ground_truth'][0][map_gt[filename]]
                ttmp = tmp['topk_score_info']
                topk_score_info = json.loads(ttmp)
                for k in topk_score_info:
                    cands[idx] = [k]
                    refs[idx] = [ground_truth]
                    idx += 1
            _, scores_t = b_scorer.compute_score(refs, cands)
            _, scores_tt = r_scorer.compute_score(refs, cands)
            scores_t.append(scores_tt)

            idx = 0
            avg_s = []
            max_s = []
            for j in range(len(method_list)):
                avg_s.append([])
                max_s.append([])
            for tmp in tqdm(predictions):
                avg_ts = []
                max_ts = []
                for j in range(len(method_list)):
                    avg_ts.append(0)
                    max_ts.append(0)
                sum_coef = 0
                ttmp = tmp['topk_score_info']
                topk_score_info = json.loads(ttmp)
                for j in range(len(topk_score_info)):
                    coef = 1/(j + 1)
                    sum_coef += coef
                    for k in range(len(method_list)):
                        avg_ts[k] += scores_t[k][idx] * coef
                        max_ts[k] = max(scores_t[k][idx], max_ts[k])

                    idx += 1
                for j in range(len(method_list)):
                    avg_ts[j]/= sum_coef
                    avg_s[j].append(avg_ts[j])
                    max_s[j].append(max_ts[j])
            if filename == 'forward':
                count_f = len(cands)
                for j, m_name in enumerate(method_list):
                    forward_metrics['avg_%s' % m_name] = mean(avg_s[j])
                for j, m_name in enumerate(method_list):
                    forward_metrics['max_%s' % m_name] = mean(max_s[j])


            else:
                count_b = len(cands)
                for j, m_name in enumerate(method_list):
                    backward_metrics['avg_%s' % m_name] = mean(avg_s[j])
                for j, m_name in enumerate(method_list):
                    backward_metrics['max_%s' % m_name] = mean(max_s[j])
        print()
        metrics = {k: round((forward_metrics[k] * count_f + backward_metrics[k] * count_b) / (count_b + count_f), 4) for k in forward_metrics}
        with open(path + 'bleu_rouge_metrics.json', 'w', encoding='utf-8') as writer:
            writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
            writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
            writer.write('average metrics: {}\n'.format(json.dumps(metrics)))