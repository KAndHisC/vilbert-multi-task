import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import poptorch

from vilbert.basebert import BaseBertForVLTasks
from vilbert.vilbert import VILBertForVLTasks

logger = logging.getLogger(__name__)



class PipelinedWithLossForSingleTask(nn.Module):
    def __init__(self, config, args, num_labels, task_cfg, task_id, task_dataloader = None) -> None:
        super().__init__()

        self.task_id_int = int(task_id[4:])
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        # some tasks will used task_dataloader[task_id].dataset.label2ans
        self.task_dataloader = task_dataloader

        # get data
        self.unzip_bach = None
        if task_id == "TASK4" or task_id == "TASK17":
            def batch_with_multiple_choice(batch):
                features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (batch)
                return features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id 
            self.unzip_bach = batch_with_multiple_choice
        else:
            def batch_without_multiple_choice(batch):
                features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (batch)
                return features, spatials, image_mask, question, target, input_mask, segment_ids, None, co_attention_mask, question_id 
            self.unzip_bach = batch_without_multiple_choice

        # process data
        self.process = None
        if task_cfg[task_id]["process"] in ["dialog"]:
            def dialog(features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id):
                batch_size = features.size(0)

                max_num_bbox = features.size(1)
                nround = question.size(1)
                num_options = question.size(2)
                rbatch_size = batch_size * nround
                question = question.view(rbatch_size, question.size(2), question.size(3))
                target = target.view(-1)
                input_mask = input_mask.view(
                    rbatch_size, input_mask.size(2), input_mask.size(3)
                )
                segment_ids = segment_ids.view(
                    rbatch_size, segment_ids.size(2), segment_ids.size(3)
                )
                co_attention_mask = co_attention_mask.view(
                    rbatch_size,
                    co_attention_mask.size(2),
                    co_attention_mask.size(3),
                    co_attention_mask.size(4),
                )

                features = (
                    features.unsqueeze(1)
                    .unsqueeze(1)
                    .expand(batch_size, nround, num_options, max_num_bbox, 2048)
                    .contiguous()
                    .view(-1, max_num_bbox, 2048)
                )
                spatials = (
                    spatials.unsqueeze(1)
                    .unsqueeze(1)
                    .expand(batch_size, nround, num_options, max_num_bbox, 5)
                    .contiguous()
                    .view(-1, max_num_bbox, 5)
                )
                image_mask = (
                    image_mask.unsqueeze(1)
                    .expand(batch_size, nround, num_options, max_num_bbox)
                    .contiguous()
                    .view(-1, max_num_bbox)
                )

                question = question.view(-1, question.size(2))
                input_mask = input_mask.view(-1, input_mask.size(2))
                segment_ids = segment_ids.view(-1, segment_ids.size(2))
                co_attention_mask = co_attention_mask.view(
                    -1, co_attention_mask.size(2), co_attention_mask.size(3)
                )
                batch_size = rbatch_size
                return features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id, batch_size, num_options
            self.process = dialog

        elif task_cfg[task_id]["process"] in ["expand"]:
            def expand(features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id):
                batch_size = features.size(0)
                max_num_bbox = features.size(1)
                num_options = question.size(1)
                features = (
                    features.unsqueeze(1)
                    .expand(batch_size, num_options, max_num_bbox, 2048)
                    .contiguous()
                    .view(-1, max_num_bbox, 2048)
                )
                spatials = (
                    spatials.unsqueeze(1)
                    .expand(batch_size, num_options, max_num_bbox, 5)
                    .contiguous()
                    .view(-1, max_num_bbox, 5)
                )
                image_mask = (
                    image_mask.unsqueeze(1)
                    .expand(batch_size, num_options, max_num_bbox)
                    .contiguous()
                    .view(-1, max_num_bbox)
                )
                question = question.view(-1, question.size(2))
                input_mask = input_mask.view(-1, input_mask.size(2))
                segment_ids = segment_ids.view(-1, segment_ids.size(2))
                co_attention_mask = co_attention_mask.view(
                    -1, co_attention_mask.size(2), co_attention_mask.size(3)
                )
                return features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id, batch_size, num_options 
            self.process = expand

        elif task_cfg[task_id]["process"] in ["retrieval"]:
            def retrieval(features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id):
                batch_size = features.size(0)
                
                max_num_bbox = features.size(1)
                num_options = question.size(1)
                features = features.view(-1, features.size(2), features.size(3))
                spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
                image_mask = image_mask.view(-1, image_mask.size(2))
                question = question.view(-1, question.size(2))
                input_mask = input_mask.view(-1, input_mask.size(2))
                segment_ids = segment_ids.view(-1, segment_ids.size(2))
                co_attention_mask = co_attention_mask.view(
                    -1, co_attention_mask.size(2), co_attention_mask.size(3)
                )
                return features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id, batch_size, num_options 
            self.process = retrieval

        elif task_cfg[task_id]["process"] in ["nlvr"]:
            def nlvr(features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id):
                batch_size = features.size(0)
                
                max_num_bbox = features.size(1)
                num_options = question.size(1)
                features = features.view(
                    batch_size * 2, int(features.size(1) / 2), features.size(2)
                )
                spatials = spatials.view(
                    batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
                )
                image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
                question = question.repeat(1, 2)
                question = question.view(batch_size * 2, int(question.size(1) / 2))
                input_mask = input_mask.repeat(1, 2)
                input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
                segment_ids = segment_ids.repeat(1, 2)
                segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
                co_attention_mask = co_attention_mask.view(
                    batch_size * 2,
                    int(co_attention_mask.size(1) / 2),
                    co_attention_mask.size(2),
                )
                return features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id, batch_size, num_options 
            self.process = nlvr
        else:
            def normal(features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id):
                batch_size = features.size(0)
                return features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id, batch_size, None 
            self.process = normal

        # get model
        if args.baseline:
            self.model = BaseBertForVLTasks.from_pretrained(
                args.from_pretrained,
                config=config,
                num_labels=num_labels,
                default_gpu=True,
            )
        else:
            self.model = VILBertForVLTasks.from_pretrained(
                args.from_pretrained,
                config=config,
                num_labels=num_labels,
                default_gpu=True,
            )
        
        # get loss
        loss_map = {
                        "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
                        "CrossEntropyLoss": nn.CrossEntropyLoss(),
                    }
        loss_func = loss_map[task_cfg[task_id]["loss"]]
        print("task %s loss is %s"%(task_id, task_cfg[task_id]["loss"]))
        self.loss = None
        if task_cfg[task_id]["type"] == "VL-classifier":
            def VL_classifier(vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_logit, question_id, batch_size, num_options, multiple_choice_ids, target):
                # results for evalution
                results = []
                loss = loss_func(vil_prediction, target)
                loss = loss.mean() * target.size(1)
                batch_score = self.compute_score_with_logits(vil_prediction, target).sum() 
                if not self.training:
                    logits = torch.max(vil_prediction, 1)[1].data  # argmax
                    for i in range(logits.size(0)):
                        results.append(
                            {
                                "question_id": question_id[i].item(),
                                "answer": self.task_dataloader.dataset.label2ans[
                                    logits[i].item()
                                ],
                            }
                        )
                return  batch_score, results, loss
            self.loss = VL_classifier

        elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
            def VL_classifier_GQA(vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_logit, question_id, batch_size, num_options, multiple_choice_ids, target):
                # results for evalution
                results = []

                loss = loss_func(vil_prediction_gqa, target)
                loss = loss.mean() * target.size(1)
                batch_score = self.compute_score_with_logits(
                    vil_prediction_gqa, target
                ).sum() 
                if not self.training:
                    logits = torch.max(vil_prediction_gqa, 1)[1].data
                    for i in range(logits.size(0)):
                        results.append(
                            {
                                "questionId": str(question_id[i].item()),
                                "prediction": self.task_dataloader.dataset.label2ans[
                                    logits[i].item()
                                ],
                            }
                        )
                return  batch_score, results, loss
            self.loss = VL_classifier_GQA

        elif task_cfg[task_id]["type"] == "VL-logit":
            def VL_logit(vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_logit, question_id, batch_size, num_options, multiple_choice_ids, target):
                # results for evalution
                results = []

                vil_logit = vil_logit.view(batch_size, num_options)
                loss = loss_func(vil_logit, target)
                _, preds = torch.max(vil_logit, 1)
                # batch_score = float((preds == target).sum()) 
                batch_score = (preds == target).sum() 
                probs = torch.softmax(vil_logit, dim=1)
                if not self.training:
                    for i in range(vil_logit.size(0)):
                        results.append(
                            {
                                "question_id": question_id[i].item(),
                                "answer": [prob.item() for prob in probs[i]],
                            }
                        )
                return  batch_score, results, loss
            self.loss = VL_logit

        elif task_cfg[task_id]["type"] == "V-logit":
            def V_logit(vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_logit, question_id, batch_size, num_options, multiple_choice_ids, target):
                # results for evalution
                results = []

                loss = loss_func(vision_logit, target)
                loss = loss.mean() * target.size(1)
                _, select_idx = torch.max(vision_logit, dim=1)
                select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
                batch_score = torch.sum(select_target > 0.5)
                if not self.training:
                    batch_score =  batch_score.item()
                    for i in range(select_idx.size(0)):
                        results.append(
                            {
                                "id": question_id[i].item(),
                                "target": select_idx[i].item(),
                                "IOU": select_target[i].item(),
                            }
                        )
                return  batch_score, results, loss
            self.loss = V_logit

        elif task_cfg[task_id]["type"] == "V-logit-mc":
            def V_logit_mc(vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_logit, question_id, batch_size, num_options, multiple_choice_ids, target):
                # results for evalution
                results = []

                vision_logit = vision_logit[:, 101:]
                vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
                vision_logit = vision_logit.unsqueeze(2)
                loss = loss_func(vision_logit, target)
                loss = loss.mean() * target.size(1)
                _, preds = torch.max(vision_logit, dim=1)
                _, target = torch.max(target, dim=1)
                # batch_score = float((preds == target).sum())
                batch_score = (preds == target).sum()
                if not self.training:
                    for i in range(preds.size(0)):
                        results.append({"id": question_id[i].item(), "target": preds[i].item()})
                return  batch_score, results, loss
            self.loss = V_logit_mc

        elif self.task_cfg[task_id]["type"] == "VL-binary-classifier":
            def VL_binary_classifier(vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_logit, question_id, batch_size, num_options, multiple_choice_ids, target):
                # results for evalution
                results = []

                loss = loss_func(vil_binary_prediction, target)
                loss = loss.mean()
                batch_score = self.compute_score_with_logits(
                    vil_binary_prediction, target
                ).sum()
                return  batch_score, results, loss
            self.loss = VL_binary_classifier

        elif self.task_cfg[task_id]["type"] == "VL-tri-classifier":
            def VL_tri_classifier(vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_logit, question_id, batch_size, num_options, multiple_choice_ids, target):
                # results for evalution
                results = []

                loss = loss_func(vil_tri_prediction, target)
                loss = loss.mean()
                batch_score = self.compute_score_with_logits(
                    vil_tri_prediction, target
                ).sum() 
                return  batch_score, results, loss
            self.loss = VL_tri_classifier  


    def forward(
        self,
        batch
    ):
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = self.unzip_bach(
            batch
        )

        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id, batch_size, num_options = self.process(
            features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id
        )
        
        
        
        
        # resize_ can't be represented in the JIT at the moment, 
        # so we won't connect any uses of this value with its current trace. 
        # If you happen to use it again, it will show up as a constant in the graph.
        # task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
        # print('question size:', question.size(), 'new size',(question.size(0), 1))
        task_tokens = torch.full( (question.size(0), 1), self.task_id_int , dtype=question.dtype)
        
        # some of them not used
        # vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, all_attention_mask 
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, _, vision_logit, _, _, _ = self.model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )

        batch_score, results, loss = self.loss(
            vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_logit, question_id, batch_size, num_options, multiple_choice_ids, target
            )


        if self.training:
            # batch_score = batch_score / float( batch_size)
            batch_score = batch_score /  batch_size
            # loss = loss * self.loss_scale[task_id]
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            return batch_score, poptorch.identity_loss(loss, reduction='none')
        else:
            # return float(batch_score), batch_size, results, float(loss)
            return  batch_score, batch_size, results, loss
    
    def compute_score_with_logits(self, logits, labels):
        logits = torch.max(logits, 1)[1].data  # argmax
        # one_hots = torch.zeros(*labels.size()).cuda()
        one_hots = torch.zeros(*labels.size())
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * labels
        return scores
