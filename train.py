# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 train.py   -run_type train   -task e2e   -max_to_keep_ckpt 60   -learning_rate 1.5e-4   -model_dir ./test/test_e2e_cpt_2  -batch_size 8   -epochs 600   -backbone fnlp/cpt-base
# CUDA_VISIBLE_DEVICES=4 python train.py   -run_type train   -task e2e   -max_to_keep_ckpt 60   -learning_rate 1.5e-4   -model_dir ./test/test_e2e_cpt_2  -batch_size 8   -epochs 600   -backbone fnlp/cpt-base


import copy
import torch
import random
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, BartForConditionalGeneration, \
    MBartForConditionalGeneration
from modeling_cpt import CPTForConditionalGeneration
from reader import CrossWOZReader, CrossWOZIterator
from evaluator import CrossWOZEvaluator
from utils import definitions_cw
from config import get_config
from fastNLP.core.log import logger
from fastNLP import Trainer,Evaluator
from fastNLP.core.callbacks import LRSchedCallback,LoadBestModelCallback,TorchGradClipCallback
from fnlp_metrics import LossMetric, DialogueMetric
from utils.fnlp_utils import get_dataset
from fastNLP.core.dataloaders.torch_dataloader import TorchDataLoader
from fastNLP.core.samplers import BucketedBatchSampler, SequentialSampler
# from utils.fnlp_utils import EvaluateSaveCallback,EvaluateCallback
from fastNLP.core.dataset import DataSet, Instance
from fastNLP.core.callbacks import MoreEvaluateCallback


cfg = get_config()

if cfg.seed > 0:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    logger.info("Set random seed to %d", cfg.seed)


def load_model(folder=None):
    if folder is not None:
        model_path = folder
    elif cfg.ckpt is not None:
        model_path = cfg.ckpt
    elif cfg.train_from is not None:
        model_path = cfg.train_from
    else:
        model_path = cfg.backbone
    logger.info('Load model from {}'.format(model_path))

    if cfg.backbone in ['fnlp/cpt-base', 'fnlp/cpt-large']:
        model = CPTForConditionalGeneration.from_pretrained(model_path)
    elif cfg.backbone in ['fnlp/bart-base-chinese', 'fnlp/bart-large-chinese']:
        model = BartForConditionalGeneration.from_pretrained(model_path)
    elif cfg.backbone in ['facebook/mbart-large-50']:
        model = MBartForConditionalGeneration.from_pretrained(model_path)
    else:
        raise NotImplementedError

    model.resize_token_embeddings(reader.vocab_size)
    return model


def get_optimizer_and_scheduler(model, num_training_steps_per_epoch, cfg):
    '''
    num_train_steps = (num_train_examples *
        self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
    '''
    num_train_steps = (num_training_steps_per_epoch * cfg.epochs) // cfg.grad_accum_steps

    if cfg.warmup_steps >= 0:
        num_warmup_steps = cfg.warmup_steps
    else:
        num_warmup_steps = int(num_train_steps * cfg.warmup_ratio)

    logger.info("Total training steps = {}, warmup steps = {}".format(
        num_train_steps, num_warmup_steps))

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    if cfg.no_learning_rate_decay:
        scheduler = get_constant_schedule(optimizer)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps)

    return optimizer, scheduler


class FModel(torch.nn.Module):
    def __init__(self, model, cfg, reader, iterator):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.reader = reader
        self.iterator = iterator
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def forward(self, belief_inputs, belief_labels, resp_inputs, resp_labels):
        attention_mask = torch.where(belief_inputs == self.reader.pad_token_id, 0, 1)
        belief_outputs = self.model(
            input_ids=belief_inputs,
            attention_mask=attention_mask,
            labels=belief_labels,
        )
        belief_loss = belief_outputs[0]
        belief_logits = belief_outputs[1]
        belief_pred = torch.argmax(belief_logits, dim=-1)

        if self.cfg.task == "e2e":
            attention_mask = torch.where(resp_inputs == self.reader.pad_token_id, 0, 1)
            resp_outputs = self.model(
                input_ids=resp_inputs,
                attention_mask=attention_mask,
                labels=resp_labels,
            )
            resp_loss = resp_outputs[0]
            resp_logits = resp_outputs[1]
            resp_pred = torch.argmax(resp_logits, dim=-1)

            num_resp_correct, num_resp_count = self.count_tokens(
                resp_pred, resp_labels, pad_id=self.reader.pad_token_id)
        num_belief_correct, num_belief_count = self.count_tokens(belief_pred, belief_labels,
                                                                 pad_id=self.reader.pad_token_id)

        if num_belief_correct > num_belief_count:
            raise Exception('acc calculating error')

        loss = belief_loss
        if self.cfg.task == "e2e" and self.cfg.resp_loss_coeff > 0:
            loss += (self.cfg.resp_loss_coeff * resp_loss)

        step_outputs = {
            'belief': {
                'loss': belief_loss.item(),
                'correct': num_belief_correct.item(),
                'count': num_belief_count.item(),
            }
        }

        if self.cfg.task == "e2e":
            step_outputs["resp"] = {"loss": resp_loss.item(),
                                    "correct": num_resp_correct.item(),
                                    "count": num_resp_count.item()}

        return {'loss': loss, "step_outputs": step_outputs}

    def count_tokens(self, pred, label, pad_id):
        num_count = label.view(-1).ne(pad_id).long().sum()
        num_correct = 0
        for i in range(label.shape[0]):
            one_pred = pred[i]
            one_label = label[i]
            valid_len = one_label.ne(pad_id).long().sum()
            one_pred = one_pred[:valid_len]
            one_label = one_label[:valid_len]
            num_correct += (one_pred == one_label).sum()

        return num_correct, num_count

    def evaluate_step(self, dial_batch):
        assert len(dial_batch)==1, "Only batch size 1 should be allowed."
        dial_batch = dial_batch[0]
        batch_size = len(dial_batch)
        dial_history = [[] for _ in range(batch_size)]

        for turn_batch in self.iterator.transpose_batch(dial_batch):
            batch_encoder_input_ids = []
            batch_encoder_resp_input_ids = []
            for t, turn in enumerate(turn_batch):
                # context = self.iterator.flatten_dial_history(
                #     dial_history[t], len(turn['user']), self.cfg.context_size
                # )
                # encoder_input_ids = [self.reader.cls_token_id] + context + turn['user'] + [self.reader.eos_token_id]

                context_belief = self.iterator.flatten_dial_history(dial_history[t], len(turn['user']),
                                                                    self.cfg.context_size)
                encoder_input_ids = [self.reader.cls_token_id] + context_belief + turn['user'] + [
                    self.reader.eos_token_id]
                batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

            batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                   batch_first=True,
                                                   padding_value=self.reader.pad_token_id)

            batch_encoder_input_ids = batch_encoder_input_ids.to(self.device)

            attention_mask = torch.where(batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

            # belief tracking
            with torch.no_grad():
                belief_outputs = self.model.generate(
                    input_ids=batch_encoder_input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=self.reader.eos_token_id,
                    max_length=200,
                    num_beams=self.cfg.beam_size,
                )

            belief_outputs = belief_outputs.cpu().numpy().tolist()
            decoded_belief_outputs = self.finalize_bspn(belief_outputs)

            for t, turn in enumerate(turn_batch):
                turn.update(**decoded_belief_outputs[t])

            if self.cfg.task == "e2e":
                dbpn = []

                if self.cfg.use_true_dbpn:
                    for turn in turn_batch:
                        dbpn.append(turn["dbpn"])
                else:
                    for turn in turn_batch:
                        bspn_gen = turn['bspn_gen']
                        bspn_gen = self.reader.tokenizer.decode(
                            bspn_gen, clean_up_tokenization_spaces=False)
                        db_token = self.reader.bspn_to_db_pointer(bspn_gen,
                                                                  turn["turn_domain"])
                        dbpn_gen = self.reader.encode_text(
                            db_token,
                            bos_token=definitions_cw.BOS_DB_TOKEN,
                            eos_token=definitions_cw.EOS_DB_TOKEN)

                        turn["dbpn_gen"] = dbpn_gen
                        dbpn.append(dbpn_gen)

                for t, turn in enumerate(turn_batch):
                    context_resp = self.iterator.flatten_dial_history(dial_history[t],
                                                                      len(turn['user']) + len(turn['bspn']) + len(
                                                                          turn['dbpn_gen']),
                                                                      self.cfg.context_size)
                    encoder_resp_input_ids = [self.reader.cls_token_id] + context_resp + turn['user'] + \
                                             turn['bspn'] + turn['dbpn_gen'] + [self.reader.eos_token_id]
                    batch_encoder_resp_input_ids.append(self.iterator.tensorize(encoder_resp_input_ids))

                batch_encoder_resp_input_ids = pad_sequence(batch_encoder_resp_input_ids,
                                                            batch_first=True,
                                                            padding_value=self.reader.pad_token_id)

                batch_encoder_resp_input_ids = batch_encoder_resp_input_ids.to(self.device)

                if self.cfg.use_true_curr_aspn:
                    # TODO 这里有 bug
                    for t, _dbpn in enumerate():
                        attention_mask = torch.where(batch_encoder_resp_input_ids == self.reader.pad_token_id, 0, 1)

                        with torch.no_grad():
                            resp_outputs = self.model.generate(
                                input_ids=batch_encoder_resp_input_ids,
                                attention_mask=attention_mask,
                                eos_token_id=self.reader.eos_token_id,
                                max_length=300,
                                num_beams=self.cfg.beam_size,
                            )
                            resp_outputs = resp_outputs.cpu().numpy().tolist()
                            decoded_resp_outputs = self.finalize_resp(resp_outputs)
                            turn_batch[t].update(**decoded_resp_outputs[0])

                else:
                    with torch.no_grad():
                        attention_mask = torch.where(batch_encoder_resp_input_ids == self.reader.pad_token_id, 0, 1)

                        resp_outputs = self.model.generate(
                            input_ids=batch_encoder_resp_input_ids,
                            attention_mask=attention_mask,
                            eos_token_id=self.reader.eos_token_id,
                            max_length=300,
                            num_beams=self.cfg.beam_size,
                        )

                    resp_outputs = resp_outputs.cpu().numpy().tolist()
                    decoded_resp_outputs = self.finalize_resp(resp_outputs)

                    for t, turn in enumerate(turn_batch):
                        turn.update(decoded_resp_outputs[t])

            # update dial_history
            for t, turn in enumerate(turn_batch):
                pv_text = copy.copy(turn["user"])

                # use true previous belief states and ignore the db stats
                if self.cfg.use_true_prev_bspn:
                    pv_bspn = turn["bspn"]
                else:
                    pv_bspn = turn["bspn_gen"]

                # use true previous response
                if self.cfg.use_true_prev_resp:
                    if self.cfg.task == "e2e":
                        pv_resp = turn["redx"]
                    else:
                        pv_resp = turn["resp"]
                else:
                    pv_resp = turn["resp_gen"]

                if self.cfg.use_true_dbpn:
                    pv_dbpn = turn["dbpn"]
                else:
                    pv_dbpn = turn['dbpn_gen']

                if self.cfg.use_true_prev_aspn:
                    pv_aspn = turn["aspn"]
                else:
                    pv_aspn = turn["aspn_gen"]

                if self.cfg.ururu:
                    pv_text += + pv_resp
                else:
                    pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                dial_history[t] = [pv_text]
        result = self.iterator.get_readable_batch(dial_batch)
        return {'result': result}

    def finalize_bspn(self, belief_outputs):
        eos_token_id = self.reader.get_token_id(definitions_cw.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}
            decoded['bspn_gen'] = bspn
            batch_decoded.append(decoded)
        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions_cw.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions_cw.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions_cw.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions_cw.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                # logger.warn("bos/eos action token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                # logger.warn("bos/eos resp token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded


reader = CrossWOZReader(cfg)
iterator = CrossWOZIterator(reader)

# num_training_steps_per_epoch 这个是将turn展开后的step
train_batches, num_training_steps_per_epoch, _, _ = iterator.get_batches('train', cfg.batch_size, shuffle=False,
                                                                         num_dialogs=cfg.num_train_dialogs,
                                                                         special_domain=cfg.special_domain)
# 先榨干获取sample，List[[(belief_inputs, resp_inputs), (belief_labels, _)], ...]
train_ds = get_dataset(train_batches, cfg.task, cfg.ururu, cfg.context_size,
                       max_seq_len=reader.max_seq_len,
                       cls_token_id=reader.cls_token_id, eos_token_id=reader.eos_token_id,
                       pad_token_id=reader.pad_token_id)

val_batches, _, _, _ = iterator.get_batches('val', cfg.batch_size, shuffle=False, num_dialogs=cfg.num_train_dialogs,
                                            special_domain=cfg.special_domain)
val_ds = get_dataset(val_batches, cfg.task, cfg.ururu, cfg.context_size,
                     max_seq_len=reader.max_seq_len,
                     cls_token_id=reader.cls_token_id, eos_token_id=reader.eos_token_id,
                     pad_token_id=reader.pad_token_id)

test_batches, _, _, _ = iterator.get_batches(cfg.pred_data_type, cfg.batch_size, special_domain=cfg.special_domain)

test_ds = DataSet()
for batch in test_batches:
    instance = Instance(dial_batch=batch)
    test_ds.append(instance)
test_ds.set_input('dial_batch')
test_ds.set_pad_val('dial_batch', val=None)


tr_dl = TorchDataLoader(dataset=train_ds,
                        batch_sampler=BucketedBatchSampler(dataset=train_ds, length='belief_inputs',
                                                           batch_size=cfg.batch_size, num_batch_per_bucket=50),
                        # sampler=RandomSampler(dataset=train_ds),
                        #     batch_size=cfg.batch_size,
                        num_workers=4)

val_dl = TorchDataLoader(dataset=val_ds,
                         batch_sampler=BucketedBatchSampler(dataset=val_ds, length='belief_inputs',
                                                            batch_size=cfg.batch_size, num_batch_per_bucket=50),
                         # sampler=RandomSampler(dataset=train_ds),
                         #     batch_size=cfg.batch_size,
                         num_workers=4)
test_dl = TorchDataLoader(dataset=test_ds,
                         # batch_sampler=BucketedBatchSampler(dataset=val_ds, length='belief_inputs',
                         #                                    batch_size=cfg.batch_size, num_batch_per_bucket=50),
                         sampler=SequentialSampler(dataset=test_ds),
                         batch_size=1,  # 必须是 1
                         num_workers=0)


ptr_model = load_model()
model = FModel(ptr_model, cfg, reader, iterator)
# cfg.learning_rate = 5e-5
optimizer, scheduler = get_optimizer_and_scheduler(model, len(tr_dl), cfg)

evaluator = CrossWOZEvaluator(reader, cfg.pred_data_type)

from fastNLP.core.callbacks import Events


@Trainer.on(Events.on_before_backward)
def show_outputs(trainer, outputs):
    logger.info(outputs['step_outputs'])


# @Trainer.on(Events.ON_AFTER_OPTIMIZERS_STEP(every=1))
# def clip_gradient(trainer, optimizers):
#     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)


def evaluate_every(trainer):
    if trainer.cur_epoch_idx > 0 and trainer.global_forward_batches % trainer.num_batches_per_epoch==0:
        return True
    return False


callbacks = [
    # LoadBestModelCallback()
    LRSchedCallback(scheduler=scheduler),
    TorchGradClipCallback(clip_value=cfg.max_grad_norm, clip_type='norm'),
    MoreEvaluateCallback(test_dl, metrics={'gen':DialogueMetric(cfg, evaluator)}, topk=10,
                         folder=cfg.model_dir, topk_monitor='combined_score#gen',
                         evaluate_every=evaluate_every)
]




trainer = Trainer(
    model=model,
    driver='torch',
    device=0,
    n_epochs=cfg.epochs,
    optimizers=optimizer,
    train_dataloader=tr_dl,
    batch_step_fn=None,
    evaluate_fn='forward',
    evaluate_dataloaders=val_dl,
    callbacks=callbacks,
    validate_every=-1,
    metrics={
        # 'dialogue': DialogueMetric(cfg, evaluator),
        'loss': LossMetric()
    },
    accumulation_steps=cfg.grad_accum_steps,
    fp16=True,
    monitor='loss',
    larger_better=False,
    torch_ddp_kwargs={'find_unused_parameters': True},
)
if cfg.run_type == 'train':
    trainer.run()
elif cfg.run_type == 'predict':
    trainer.load_model(
        folder = cfg.model_dir,
        only_state_dict=True
    )
    evaluator = Evaluator(
        model=model,
        driver=trainer.driver,
        dataloaders=test_dl,
        metrics={'gen':DialogueMetric(cfg, evaluator)},
        fp16=True
    )

    evaluator.run()


