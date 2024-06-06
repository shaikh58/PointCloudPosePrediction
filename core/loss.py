import torch


def PoseLoss(pred, target):
    """Expects a length 7 vector; upper 3 elements are relative translation, 
    lower 4 are relative rotation in quaternion form"""
    
    pos_loss = torch.linalg.norm(target[:,:3] - pred[:,:3])
    # normalize predicted rotation vector to ensure valid quaternion; clamp to avoid zero div
    q_norm = torch.linalg.norm(pred[:,3:], dim=1).view(-1,1)
    rot_loss = torch.linalg.norm(target[:,3:] - (pred[:,3:]/torch.clamp(q_norm, min=1e-5)))
    
    return pos_loss + rot_loss