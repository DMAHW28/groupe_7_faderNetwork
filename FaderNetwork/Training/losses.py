import torch.nn as nn

def adversarial_loss(pred, target):
    return nn.BCELoss()(pred, target)

def reconstruction_loss(recon, original):
    return nn.MSELoss()(recon, original)

def attribute_loss(pred_attributes, target_attributes):
    return nn.BCELoss()(pred_attributes, target_attributes)
