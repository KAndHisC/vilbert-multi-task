
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import poptorch



# from vilbert.basebert import BaseBertForVLTasks
from vilbert.vilbert import VILBertForVLTasks

logger = logging.getLogger(__name__)

class RecomputationCheckpoint(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return poptorch.recomputationCheckpoint(self.layer(x))
        # return tuple(poptorch.recomputationCheckpoint(y) for y in self.layer(x))

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
        # print(self.model)
        # for name, p in self.model.named_parameters():
        #     # if p.requires_grad:
        #     print(name, p.numel()/1000000, p.dtype) 
        # exit()

        
        # 0
        self.model.bert.embeddings = poptorch.BeginBlock(self.model.bert.embeddings, "embeddings", ipu_id=0) # 24m
        self.model.bert.v_embeddings = poptorch.BeginBlock(self.model.bert.v_embeddings, "v_embeddings", ipu_id=0) # 2m

        # self.model.bert.encoder = poptorch.BeginBlock(RecomputationCheckpoint(self.model.bert.encoder) , "encoder", ipu_id=1)
        # "v_biattention_id":[0, 1],
        # "t_biattention_id":[10, 11],
        # layer 7m * 12 + v_layer 6m*6 + c_layer 17m*6
        # t 12 v 2 c 3*2  =  4 20 4  7

        c_layer_length = len(self.model.bert.encoder.c_layer)
        t_layer_length = len(self.model.bert.encoder.layer)
        offset = t_layer_length - c_layer_length

        # t_layer 0-10 
        layers_on_ipu = [0,0,0,1,1,1,1,1,1,1]
        for index in range(offset):
            layer = self.model.bert.encoder.layer[index]
            # layer = RecomputationCheckpoint(layer) 
            self.model.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"t_layer{index}", ipu_id=layers_on_ipu[index])
            print(f"layer {index:<2} --> IPU {layers_on_ipu[index]}") 

        # c t v 3*c_layer_length
        layers_on_ipu = [2,2,2,2,3,3]
        count = 0
        for index in range(c_layer_length):
            c_layer = self.model.bert.encoder.c_layer[index]
            # c_layer = RecomputationCheckpoint(c_layer) 
            self.model.bert.encoder.c_layer[index] = poptorch.BeginBlock(c_layer, f"c_layer{index}", ipu_id=layers_on_ipu[count])
            print(f"c_layer {index:<2} --> IPU {layers_on_ipu[count]}") # layer 7m * 12 + v_layer 6m*6 + c_layer 17m*6
            count += 1

            v_layer = self.model.bert.encoder.v_layer[index]
            # v_layer = RecomputationCheckpoint(v_layer) 
            self.model.bert.encoder.v_layer[index] = poptorch.BeginBlock(v_layer, f"v_layer{index}", ipu_id=layers_on_ipu[count])
            print(f"v_layer {index:<2} --> IPU {layers_on_ipu[count]}") 
            count += 1

            t_index = index+offset
            t_layer = self.model.bert.encoder.layer[t_index]
            # t_layer = RecomputationCheckpoint(t_layer) 
            self.model.bert.encoder.layer[t_index] = poptorch.BeginBlock(t_layer, f"t_layer{t_index}", ipu_id=layers_on_ipu[count])
            print(f"t_layer {t_index:<2} --> IPU {layers_on_ipu[count]}") 
            count += 1 
        
      
        # 3
        self.model.bert.t_pooler = poptorch.BeginBlock(self.model.bert.t_pooler, "t_pooler", ipu_id=3) # 1m
        self.model.bert.v_pooler = poptorch.BeginBlock(self.model.bert.v_pooler, "v_pooler", ipu_id=3) # 1m
        self.model.cls = poptorch.BeginBlock(self.model.cls, "cls", ipu_id=3) # 3m

        self.model.dropout = poptorch.BeginBlock(self.model.dropout, "dropout", ipu_id=3)
        
        self.model.vil_prediction = poptorch.BeginBlock(self.model.vil_prediction, "vil_prediction", ipu_id=3) # 8m
        self.model.vil_prediction_gqa = poptorch.BeginBlock(self.model.vil_prediction_gqa, "vil_prediction_gqa", ipu_id=3) # 5m
        self.model.vil_binary_prediction = poptorch.BeginBlock(self.model.vil_binary_prediction, "vil_binary_prediction", ipu_id=3) # 4m

        self.model.vil_logit = poptorch.BeginBlock(self.model.vil_logit, "vil_logit", ipu_id=3)

        self.model.vil_tri_prediction = poptorch.BeginBlock(self.model.vil_tri_prediction, "vil_tri_prediction", ipu_id=3)
        self.model.vision_logit = poptorch.BeginBlock(self.model.vision_logit, "vision_logit", ipu_id=3)
        self.model.linguisic_logit = poptorch.BeginBlock(self.model.linguisic_logit, "linguisic_logit", ipu_id=3)
        
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
        batch_score = preds.eq(target).sum().float()

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
