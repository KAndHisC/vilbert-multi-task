
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import poptorch


# from vilbert.basebert import BaseBertForVLTasks
from vilbert.vilbert import VILBertForVLTasks

logger = logging.getLogger(__name__)

class PipelinedWithLossForRetrievalFlickr30k(nn.Module):
    def __init__(self, config, args, num_labels):
        super().__init__()

        # self.task_id_int = int(task_id[4:])
        # self.gradient_accumulation_steps = args.gradient_accumulation_steps
        # some tasks will used task_dataloader[task_id].dataset.label2ans

        # get model
        # if args.baseline:
        #     self.model = BaseBertForVLTasks.from_pretrained(
        #         args.from_pretrained,
        #         config=config,
        #         num_labels=num_labels,
        #         default_gpu=True,
        #     )
        # else:
        self.model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=True,
        )
        # # 0
        # poptorch.BeginBlock(self.model.bert.embeddings, "embeddings", ipu_id=0)
        # poptorch.BeginBlock(self.model.bert.v_embeddings, "v_embeddings", ipu_id=0)
        # # 1
        # poptorch.BeginBlock(self.model.bert.encoder, "encoder", ipu_id=1)
        # # 2
        # poptorch.BeginBlock(self.model.bert.t_pooler, "t_pooler", ipu_id=2)
        # poptorch.BeginBlock(self.model.bert.v_pooler, "v_pooler", ipu_id=2)
        # poptorch.BeginBlock(self.model.cls, "cls", ipu_id=2)
        # # 3
        # poptorch.BeginBlock(self.model.dropout, "dropout", ipu_id=3)
        # poptorch.BeginBlock(self.model.vil_prediction, "vil_prediction", ipu_id=3)
        # poptorch.BeginBlock(self.model.vil_prediction_gqa, "vil_prediction_gqa", ipu_id=3)
        # poptorch.BeginBlock(self.model.vil_binary_prediction, "vil_binary_prediction", ipu_id=3)
        # poptorch.BeginBlock(self.model.vil_logit, "vil_logit", ipu_id=3)
        # poptorch.BeginBlock(self.model.vil_tri_prediction, "vil_tri_prediction", ipu_id=3)
        # poptorch.BeginBlock(self.model.vision_logit, "vision_logit", ipu_id=3)
        # poptorch.BeginBlock(self.model.linguisic_logit, "linguisic_logit", ipu_id=3)
        
        self.loss = nn.CrossEntropyLoss()


      

    def forward(
        self,
        batch
    ):
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (batch)

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
        # resize_ can't be represented in the JIT at the moment, 
        # so we won't connect any uses of this value with its current trace. 
        # If you happen to use it again, it will show up as a constant in the graph.
        # task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
        # print(question.dtype)
        task_tokens = torch.full( (question.size(0), 1), 8 , dtype=question.dtype)
        
        # some of them not used
        # vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, all_attention_mask 
        _, _, vil_logit, _, _, _, _, _, _, _ = self.model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )

        # results for evalution
        results = []

        vil_logit = vil_logit.view(batch_size, num_options)
        loss = self.loss(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        # batch_score = float((preds == target).sum()) 
        batch_score = torch.eq(preds, target).sum().float()

        if self.training:
            # batch_score = batch_score / float( batch_size)
            batch_score = batch_score /  batch_size
            # loss = loss * self.loss_scale[task_id]
            # if self.gradient_accumulation_steps > 1:
            #     loss = loss / self.args.gradient_accumulation_steps
            return batch_score, loss
        else:
            probs = torch.softmax(vil_logit, dim=1)
            for i in range(vil_logit.size(0)):
                results.append(
                    {
                        "question_id": question_id[i].item(),
                        "answer": [prob.item() for prob in probs[i]],
                    }
                )
            # return float(batch_score), batch_size, results, float(loss)
            return  batch_score, batch_size, results, loss
    
    # def compute_score_with_logits(self, logits, labels):
    #     logits = torch.max(logits, 1)[1].data  # argmax
    #     # one_hots = torch.zeros(*labels.size()).cuda()
    #     one_hots = torch.zeros(*labels.size())
    #     one_hots.scatter_(1, logits.view(-1, 1), 1)
    #     scores = one_hots * labels
    #     return scores
