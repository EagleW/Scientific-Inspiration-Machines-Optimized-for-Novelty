
from bleu.bleu import Bleu
from rouge.rouge import Rouge
class Evaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4),  ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L")
            ]

    

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores
    def evaluate(self, quesid2ans):
        hypo = {}
        ref = {}
        i = 0
        for k in quesid2ans:
            ans, tgt = quesid2ans[k]
            hypo[i] = [ans]
            ref[i] = [tgt]
            i += 1

        score = self.score(ref, hypo)
        print(score)
        
        return {'score':2*score['ROUGE_L']*score['Bleu_4']/(score['Bleu_4']+ score['ROUGE_L']), 'bleu':score['Bleu_4'], 'rogue':score['ROUGE_L']}