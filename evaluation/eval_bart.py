import json
from tqdm import tqdm
from statistics import mean
from bart_score import BARTScorer

bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='bart.pth')


map_gt = {
    'forward': 'tail',
    'backward': 'head'
}
dir_names = [
    '../models/T5/%s/', 
]
model_nameses = [
    ['t5_checkpoint'],
]
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
                ground_truth = tmp['ground_truth'][0][map_gt[filename]]
                ttmp = tmp['topk_score_info']
                topk_score_info = json.loads(ttmp)
                for k in topk_score_info:
                    refs.append(ground_truth)
                    cands.append(k)
            scores = bart_scorer.score(cands, refs, batch_size=128)

            bart_f1s = []
            max_bart_f1s = []
            index = 0
            for i, tmp in enumerate(predictions):
                bart_f1 = 0
                sum_coef = 0
                ttmp = tmp['topk_score_info']
                topk_score_info = json.loads(ttmp)
                max_bart_f1 = -1000
                for j in range(len(topk_score_info)):
                    coef = 1/(j + 1)
                    sum_coef += coef
                    bart_f1 += scores[index] * coef
                    max_bart_f1 = max(max_bart_f1, scores[index])
                    index += 1
                bart_f1 /= sum_coef

                tmp['avg_bartscore'] = bart_f1
                tmp['max_bartscore'] = max_bart_f1

                bart_f1s.append(bart_f1)
                max_bart_f1s.append(max_bart_f1)
            if filename == 'forward':
                count_f = len(predictions)
                forward_metrics['avg_bartscore'] = mean(bart_f1s)
                forward_metrics['max_bartscore'] = mean(max_bart_f1s)
                with open(path + 'eval_all_forward.json', 'w', encoding='utf-8') as writer:
                    writer.write(json.dumps(predictions, ensure_ascii=False, indent=4))
            else:
                count_b = len(predictions)
                backward_metrics['avg_bartscore'] = mean(bart_f1s)
                backward_metrics['max_bartscore'] = mean(max_bart_f1s)
                with open(path + 'eval_all_backward.json', 'w', encoding='utf-8') as writer:
                    writer.write(json.dumps(predictions, ensure_ascii=False, indent=4))
        print()
        metrics = {k: round((forward_metrics[k] * count_f + backward_metrics[k] * count_b) / (count_b + count_f), 4) for k in forward_metrics}
        with open(path + 'bart_metrics.json', 'w', encoding='utf-8') as writer:
            writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
            writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
            writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

