import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def F1_score(outputs, targets, device, threshold):
    #F1 = torch.FloatTensor([0]).to(device)
    Pr = torch.FloatTensor([0]).to(device)
    Re = torch.FloatTensor([0]).to(device)
    for i, c in enumerate(zip(outputs, targets)):
        output = torch.where(c[0].view(-1) > threshold, torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
        eps = 0.00001
        inter = torch.dot(output, c[1].view(-1))
        union = torch.sum(output) + torch.sum(c[1]) + eps
        if torch.sum(c[1]).item() == 0 and torch.sum(output).item() == 0:
            #print('non_crack correct F1: {}'.format(F1))
            Pr+=torch.FloatTensor([1]).to(device)
            Re+=torch.FloatTensor([1]).to(device)
            #F1+=torch.FloatTensor([1]).to(device)
        elif torch.sum(c[1]).item() == 0 and torch.sum(output).item() != 0:
            #print('non_crack False F1: {}'.format(F1))
            Pr+=torch.FloatTensor([0]).to(device)
            Re+=torch.FloatTensor([0]).to(device)
            #F1+=torch.FloatTensor([0]).to(device)
        else:
           #print('crack correct F1: {}'.format(F1))
            Pr+=inter.float() / (torch.sum(output) + eps)
            Re+=inter.float() / (torch.sum(c[1]) + eps)
            #F1+=2 * Pr * Re / (Pr + Re + eps)
            
    Precision = Pr / (i+1)
    Recall = Re / (i+1)
    f1 = 2 * Precision*Recall / (Recall+Precision+eps)
    return f1, Precision, Recall


def dice_loss2(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )