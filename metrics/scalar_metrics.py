import torch

def binary_accuracy(prediction, target):
  """Calculates accuracy between two classes using probabilities

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Vector of probabilities of class 1
  target : torch.Tensor
    Vector containing the class indices 0 or 1
  """
  if isinstance(prediction, torch.autograd.Variable):
      prediction = prediction.data
  predicted_classes = torch.gt(prediction, 0.5)
  num_correct = torch.sum(torch.eq(predicted_classes, target.byte()))
  return num_correct / prediction.numel()
 
