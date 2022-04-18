from itertools import chain
import torch

from fastNLP.core.dataset import Instance, DataSet
from fastNLP.core.utils import f_rich_progress
from fastNLP import Evaluator
from fastNLP.core.callbacks.has_monitor_callback import HasMonitorCallback
from fastNLP.core.log import logger

from fastNLP.core.callbacks import ModelCheckpointCallback
import json
import os


class EvaluateSaveCallback(ModelCheckpointCallback):
    def __init__(self, data, metrics, cfg,
                 monitor='bleu',
                 save_folder=None,
                 save_every_n_epochs=None,
                 save_every_n_batches=None,
                 save_last: bool = False,
                 save_topk=None,
                 save_on_exception=None,
                 larger_better: bool = True,
                 only_state_dict: bool = True,
                 model_save_fn=None,
                 **kwargs, ):
        super().__init__(monitor, save_folder, save_every_n_epochs, save_every_n_batches, save_last,
                         save_topk, save_on_exception, larger_better, only_state_dict, model_save_fn, **kwargs)
        self.data = data
        self.metrics = metrics
        self.cfg = cfg

    def on_after_trainer_initialized(self, trainer, driver):
        super().on_after_trainer_initialized(trainer, driver)
        self.evaluator = Evaluator(model=trainer.model, dataloaders=self.data, driver=driver, metrics=self.metrics)

    def on_validate_end(self, trainer, results):
        if trainer.cur_epoch_idx > 100 and self.cfg.task == 'dst':
            results = self.evaluator.run()
            folder = self._save_topk(trainer, results)
            self._save_results(results, folder)

        elif trainer.cur_epoch_idx > 10:
            results = self.evaluator.run()
            folder = self._save_topk(trainer, results)
            self._save_results(results, folder)

    def _save_results(self, results, folder):
        if folder:
            with open(os.path.join(folder, 'results.json'), 'r') as f:
                json.dump(results, f)

    def on_sanity_check_end(self, trainer, sanity_check_res):
        pass

class EvaluateCallback(HasMonitorCallback):
    def __init__(self, data, metrics, cfg, monitor=None, larger_better=True):
        super(EvaluateCallback, self).__init__(monitor, larger_better, True)
        self.data = data
        self.metrics = metrics
        self.cfg = cfg

    def on_after_trainer_initialized(self, trainer, driver):
        super(EvaluateCallback, self).on_after_trainer_initialized(trainer, driver)
        self.evaluator = Evaluator(model=trainer.model, dataloaders=self.data, driver=driver, metrics=self.metrics)

    def on_validate_end(self, trainer, results):
        # if self.is_better_results(results) and trainer.cur_epoch_idx>100 and self.cfg.task=='dst':
        if trainer.cur_epoch_idx>100 and self.cfg.task=='dst':
            self.evaluator.run()
        # elif self.is_better_results(results) and trainer.cur_epoch_idx>10:
        elif trainer.cur_epoch_idx>10:
            self.evaluator.run()


def flatten_dial_history(dial_history, len_postfix, context_size, max_seq_len):
    if context_size > 0:
        context_size -= 1

    if context_size == 0:
        windowed_context = []
    elif context_size > 0:
        windowed_context = dial_history[-context_size:]
    else:
        windowed_context = dial_history

    ctx_len = sum([len(c) for c in windowed_context])

    spare_len = max_seq_len - len_postfix - 2
    while ctx_len >= spare_len:
        ctx_len -= len(windowed_context[0])
        windowed_context.pop(0)

    context = list(chain(*windowed_context))

    return context


def transpose_batch(dial_batch):
    turn_batch = []
    turn_num = len(dial_batch[0])
    for turn in range(turn_num):
        turn_l = []
        for dial in dial_batch:
            this_turn = dial[turn]
            turn_l.append(this_turn)
        turn_batch.append(turn_l)
    return turn_batch


def tensorize(ids):
    return torch.tensor(ids, dtype=torch.long)


def get_dataset(all_batches, task, ururu, context_size=-1, max_seq_len=512,
                cls_token_id=0, eos_token_id=1, pad_token_id=0):
    ds = DataSet()
    # dial_progress = f_rich_progress.add_task('dialogue', total=len(all_batches))
    for dial_idx, dial_batch in enumerate(all_batches):
        batch_encoder_input_ids = []
        batch_encoder_resp_inputs_ids = []
        batch_belief_label_ids = []
        batch_resp_label_ids = []
        # batch_whole_label_ids = []

        for _, dial in enumerate(dial_batch):
            dial_encoder_inputs_ids = []
            dial_beleif_label_ids = []
            dial_resp_label_ids = []
            dial_resp_inputs_ids = []
            # dial_whole_label_ids =[]

            dial_history = []
            for turn in dial:
                context_belief = flatten_dial_history(dial_history, len(turn['user']), context_size, max_seq_len)
                context_resp = flatten_dial_history(dial_history
                                                         ,len(turn['user'] ) +len(turn['bspn'] ) +len(turn['dbpn'])
                                                         ,context_size, max_seq_len)
                encoder_input_ids = [cls_token_id] + context_belief + turn['user'] + \
                    [eos_token_id]
                encoder_resp_input_ids = [cls_token_id] + context_resp + turn['user'] + turn['bspn'] + turn[
                    'dbpn'] + [eos_token_id]
                # belief labels
                bspn = turn['bspn']
                bspn_label = bspn
                belief_label_ids = bspn_label + [eos_token_id]
                # resp labels
                resp = turn["aspn"] + turn["redx"]
                resp_label_ids = resp + [eos_token_id]

                # whole_label_ids = bspn_label + resp + [self.reader.eos_token_id]

                dial_encoder_inputs_ids.append(encoder_input_ids)
                dial_resp_inputs_ids.append(encoder_resp_input_ids)

                dial_beleif_label_ids.append(belief_label_ids)
                dial_resp_label_ids.append(resp_label_ids)
                # dial_whole_label_ids.append(whole_label_ids)

                if ururu:
                    if task == "dst":
                        turn_text = turn["user"] + turn["resp"]
                    else:
                        turn_text = turn["user"] + turn["redx"]
                else:
                    if task == "dst":
                        turn_text = turn["user"] + bspn + \
                                    turn["dbpn"] + turn["aspn"] + turn["resp"]
                    else:
                        turn_text = turn["user"] + bspn + \
                                    turn["dbpn"] + turn["aspn"] + turn["redx"]

                dial_history.append(turn_text)

            batch_encoder_input_ids.append(dial_encoder_inputs_ids)
            batch_encoder_resp_inputs_ids.append(dial_resp_inputs_ids)
            batch_belief_label_ids.append(dial_beleif_label_ids)
            batch_resp_label_ids.append(dial_resp_label_ids)
            # batch_whole_label_ids.append(dial_whole_label_ids)

        # turn first
        batch_encoder_input_ids = transpose_batch(batch_encoder_input_ids)
        batch_encoder_resp_inputs_ids = transpose_batch(batch_encoder_resp_inputs_ids)
        batch_belief_label_ids = transpose_batch(batch_belief_label_ids)
        batch_resp_label_ids = transpose_batch(batch_resp_label_ids)
        # batch_whole_label_ids = self.transpose_batch(batch_whole_label_ids)

        num_turns = len(batch_belief_label_ids)

        # tensor_whole_label_ids = []
        for t in range(num_turns):
            tensor_encoder_input_ids = [b for b in batch_encoder_input_ids[t]]
            tensor_encoder_resp_inputs_ids = [b for b in batch_encoder_resp_inputs_ids[t]]
            tensor_belief_label_ids = [b for b in batch_belief_label_ids[t]]
            tensor_resp_label_ids = [b for b in batch_resp_label_ids[t]]
            # tensor_whole_label_ids = [self.tensorize(b) for b in batch_whole_label_ids[t]]
            for a,b,c,d in zip(tensor_encoder_input_ids, tensor_encoder_resp_inputs_ids, tensor_belief_label_ids, tensor_resp_label_ids):
                ins = Instance(dial_idx=dial_idx, turn_idx=t,
                               belief_inputs=a,
                               resp_inputs=b,
                               belief_labels=c,
                               resp_labels=d)
                ds.append(ins)
    #     f_rich_progress.update(dial_progress, advance=1, refresh=True)
    # f_rich_progress.destroy_task(dial_progress)
    ds.set_input('belief_inputs', 'resp_inputs', 'belief_labels', 'resp_labels')
    ds.set_pad_val('belief_inputs', 'resp_inputs', 'belief_labels', 'resp_labels', val=pad_token_id)
    return ds


