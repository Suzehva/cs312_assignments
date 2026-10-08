"""P4.2 policies, preserving the course model and trainer conventions.

Width muP changes exactly the four settings in the handout. Depth variants
add residual, block-LR and block-epsilon corrections at reference depth eight.
Builder arguments are saved by the shared trainer for checkpoint reconstruction.
"""
import math
import torch
from torch.nn import functional as F
from modeling import AutoregressiveLM, LlamaDecoderLayer, initialize_model
from optimizers import build_masked_weight_decay_parameter_groups


class ScaledDecoderLayer(LlamaDecoderLayer):
    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        branch = self.self_attn(self.input_layernorm(hidden_states),
                                position_embeddings=position_embeddings,
                                attention_mask=attention_mask)
        branch = F.dropout(branch, p=self.dropout, training=self.training)
        hidden_states = residual + self.residual_multiplier * branch
        residual = hidden_states
        branch = self.mlp(self.post_attention_layernorm(hidden_states))
        branch = F.dropout(branch, p=self.dropout, training=self.training)
        return residual + self.residual_multiplier * branch


class ParameterizedLM(AutoregressiveLM):
    def __init__(self, config, dtype=torch.float32, qk_norm=True,
                 tie_word_embeddings=False, dropout=0., prescription='mup',
                 reference_width=512, reference_depth=8):
        if prescription not in ('baseline','mup','depth_mup','completep'):
            raise ValueError(prescription)
        if tie_word_embeddings:
            raise ValueError('P4.2 requires untied embedding and readout')
        super().__init__(config,dtype=dtype,qk_norm=qk_norm,
                         tie_word_embeddings=False,dropout=dropout)
        self.prescription = prescription
        self.reference_width = reference_width
        self.reference_depth = reference_depth
        r = self.config.num_hidden_layers / reference_depth
        self.residual_multiplier = (r**-.5 if prescription=='depth_mup' else
                                    r**-1 if prescription=='completep' else 1.)
        self.output_multiplier = (1. if prescription=='baseline' else
                                  reference_width/self.config.hidden_size)
        if prescription in ('depth_mup','completep'):
            for i in range(len(self.model.layers)):
                layer = ScaledDecoderLayer(self.config,i,qk_norm=qk_norm,dropout=dropout)
                layer.residual_multiplier = self.residual_multiplier
                self.model.layers[i] = layer.to(dtype=dtype)

    def initialize_parameters(self):
        # Reusing the supplied initializer guarantees the same ordered Gaussian
        # draws (including the shared embedding/readout draw) in both policies.
        initialize_model(self)
        if self.prescription != 'baseline':
            m = self.config.hidden_size/self.reference_width
            with torch.no_grad():
                self.model.embed_tokens.weight.mul_(m)
                self.lm_head.weight.mul_(math.sqrt(m))

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        if input_ids is None:
            raise ValueError('input_ids must be provided')
        h = self.model(input_ids=input_ids,attention_mask=attention_mask,
                       position_ids=position_ids)
        return self.lm_head(h*self.output_multiplier)


def build_optimizer(model,optimizer_name,learning_rate,weight_decay,beta1,beta2):
    if optimizer_name!='adamw':
        raise ValueError('P4.2 fixes AdamW')
    hidden = {id(p) for module in model.model.layers.modules()
              if isinstance(module,torch.nn.Linear) for p in module.parameters(recurse=False)}
    block = {id(p) for p in model.model.layers.parameters()}
    m=model.config.hidden_size/model.reference_width
    r=model.config.num_hidden_layers/model.reference_depth
    block_lr=r**-.5 if model.prescription=='depth_mup' else 1.
    block_eps=(r**-.5 if model.prescription=='depth_mup' else
               r**-1 if model.prescription=='completep' else 1.)
    groups=[]
    for masked in build_masked_weight_decay_parameter_groups(model,weight_decay):
        split={}
        for p in masked['params']:
            factor=1/m if id(p) in hidden and model.prescription!='baseline' else 1.
            if id(p) in block:
                factor*=block_lr
            eps=1e-8*(block_eps if id(p) in block else 1.)
            split.setdefault((factor,eps),[]).append(p)
        for (factor,eps),params in split.items():
            groups.append(dict(params=params,lr=learning_rate*factor,
                               eps=eps,weight_decay=masked['weight_decay']))
    kwargs={'betas':(beta1,beta2),'lr':learning_rate,'weight_decay':0.}
    if any(p.is_cuda for p in model.parameters()):
        kwargs['fused']=True
    return torch.optim.AdamW(groups,**kwargs)
