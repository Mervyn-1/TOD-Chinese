from fastNLP.core.metrics import Metric
from utils.io_utils import save_json
import json
import os

class DialogueMetric(Metric):
    def __init__(self, cfg, evaluator):
        super().__init__(backend='torch')
        self.cfg = cfg
        self.evaluator = evaluator
        self.results = {}

    def update(self, result):
        self.results.update(**result)

    def reset(self):
        self.results = {}

    def get_metric(self) -> dict:
        # 适配多卡训练的场景
        result_lst = self.backend.all_gather_object(self.results)
        results = {}
        for res in result_lst:
            results.update(res)
        if self.cfg.run_type =='predict':
            save_json(results, os.path.join(self.cfg.model_dir, self.cfg.output))
        if self.cfg.task == 'dst':
            joint_goal, f1, accuracy, count_dict, correct_dict = self.evaluator.dialog_state_tracking_eval(results)
            metric_results = {'joint_acc': joint_goal, 'acc': accuracy, 'f1': f1}
            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100
                metric_results[domain_slot] = acc
        else:
            bleu, success, match = self.evaluator.e2e_eval(results, eval_dial_list=None)

            score = 0.5 * (success + match) + bleu

            metric_results = {'match': match, 'success': success, 'bleu': bleu, 'combined_score': score}
        return metric_results


class LossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.register_element('loss', aggregate_method='sum')

    def update(self, loss):
        self.loss += loss

    def get_metric(self) -> dict:
        return self.loss.item()