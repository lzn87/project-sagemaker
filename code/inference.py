DEBUG = False # false for deployment
from PIL import Image

print('PIL imported')

import numpy as np
import os
import json

import sys
if DEBUG:
    sys.path.append('pytorch-image-models')
else:
    sys.path.append('inference_model/code')
    sys.path.append('inference_model/code/pytorch-image-models')
    sys.path.append('inference_model/code/boto3')

try:
    import boto3
except Exception as e:
    print('boto3 failed')
    print('boto3 error', e)
else:
    print('boto3 imported')
    
try:
    import torchvision.transforms as transforms
except Exception as e:
    print('torchvision failed')
    print('torchvision error', e)
else:
    print('torchvision imported')

if DEBUG:
    print('DEBUG MODE')
else:
    print('pwd:', os.getcwd())
    print('ls ', os.listdir())
    print('ls code', os.listdir('code'))
    print('ls inference_model', os.listdir('inference_model'))
    print('ls inference_model/code', os.listdir('inference_model/code'))

import torch
import timm
import math
from torch import nn
import torch.nn.functional as F 

class CFG:
    img_size = 512
    batch_size = 12
    seed = 2020
    
    device = 'cuda'
    classes = 11014
    
    scale = 30 
    margin = 0.5
    
class Custom_ToTensor(object):        
    def __call__(self, sample):
        sample = np.array(sample)
        sample = torch.from_numpy(sample).float()
        sample = sample.permute((2, 0, 1)).unsqueeze(0)
        sample = nn.functional.interpolate(sample, (512, 512), mode='bilinear').squeeze()
        return sample / 255.

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

class ShopeeModel(nn.Module):

    def __init__(
        self,
        n_classes = CFG.classes,
        model_name = None,
        fc_dim = 512,
        margin = CFG.margin,
        scale = CFG.scale,
        use_fc = True,
        pretrained = False):


        super(ShopeeModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif 'efficientnet' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        
        elif model_name == 'eca_nfnet_l0':
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling =  nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            scale = scale,
            margin = margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feat(image)
        #logits = self.final(feature,label)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x
    
def replace_activations(model, existing_layer, new_layer):
    
    """A function for replacing existing activation layers"""
    
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activations(module, existing_layer, new_layer)

        if type(module) == existing_layer:
            layer_old = module
            layer_new = new_layer
            model._modules[name] = layer_new
    return model

class Mish_func(torch.autograd.Function):
    
    """from: https://github.com/tyunist/memory_efficient_mish_swish/blob/master/mish.py"""
    
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
  
        v = 1. + i.exp()
        h = v.log() 
        grad_gh = 1./h.cosh().pow_(2) 

        # Note that grad_hv * grad_vx = sigmoid(x)
        #grad_hv = 1./v  
        #grad_vx = i.exp()
        
        grad_hx = i.sigmoid()

        grad_gx = grad_gh *  grad_hx #grad_hv * grad_vx 
        
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx 
        
        return grad_output * grad_f 


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)
    

def get_model(model_name = None, model_path = None, n_classes = None):
    
    model = ShopeeModel(model_name = model_name)
    if model_name == 'eca_nfnet_l0':
        model = replace_activations(model, torch.nn.SiLU, Mish())
    model.eval()
    return model 

class EnsembleModel(nn.Module):
    
    def __init__(self):
        super(EnsembleModel,self).__init__()
        self.m1 = get_model('eca_nfnet_l0', None)
        self.m2 = get_model('tf_efficientnet_b5_ns', None)
        
    def forward(self,img, label=None):
        
        feat1 = self.m1(img, None)
        feat2 = self.m2(img, None)
        
        return torch.cat((feat1, feat2), dim=1) / 2

def model_fn(model_dir):
    print('model_dir', model_dir)
    model = EnsembleModel()
    print('model created!!')
    if DEBUG:
        model_path = os.path.join(model_dir, 'model_state_dict.pth')
    else:
        model_path = os.path.join(model_dir, 'inference_model/model_state_dict.pth')
    print('model_path', model_path)
    model.load_state_dict(torch.load(model_path))
    print('model loaded!!')
    return model


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
#     return request_body 
    print('input_fn', request_body)
    print('input_fn', request_content_type)
    json_body = json.loads(request_body)
    print('json body', json_body)
    bucket = json_body['bucket']
    key = json_body['key']
    print('bucket', bucket, 'key', key)
    
    s3_client = boto3.client('s3')
    img = s3_client.get_object(Bucket=bucket, Key=key)
    img = Image.open(img['Body'])
    
    print('image obtained from s3')
    
    transform = transforms.Compose([
        Custom_ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img = transform(img).unsqueeze(0)
    
    print('image transformed to tensor', img.size())
    
    return img

def predict_fn(input_data, model):
    model.eval()
    print('pred_fn!', 'input_data', type(input_data), input_data)
    with torch.no_grad():
        out =  model(input_data)
    print('out!', out.size())
    return out