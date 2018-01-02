from metrics import image_metrics
from metrics import scalar_metrics

MIN_VALUE = float('-inf')
MAX_VALUE = float('inf')


class Metric(object):
  def __init__(self, value):
    self.value = value
    self.num_values = value
    self.num_updates = 1

  @property
  def worst_value(self):
    """Return worst value

    Returns value which is worse than all other values according to metric
    """
    raise NotImplementedError('Subclasses must override worst_value')

  def __str__(self):
    return '{:.4f}'.format(self.value)

  def __gt__(self, other):
    """Returns true iff this metric is better than other according to metric"""
    raise NotImplementedError('Subclasses must override __gt__')

  def accumulate(self, metric):
    self.value = metric.value
    self.num_updates += 1
    self.num_values += metric.value

  def average(self):
    return type(self)(self.num_values / self.num_updates)


class MinMetric(Metric):
  """Metric for which smaller values are better"""
  def __init__(self, value):
    super(MinMetric, self).__init__(value)

  @property
  def worst_value(self):
    """Return worst value"""
    return MinMetric(MAX_VALUE)

  def __gt__(self, other):
    """Returns true iff this metric is better than other according to metric

    For MinMetric, smaller values are better
    """
    return self.value < other.value


class MaxMetric(Metric):
  """Metric for which larger values are better"""
  def __init__(self, value):
    super(MaxMetric, self).__init__(value)

  @property
  def worst_value(self):
    """Return worst value"""
    return MaxMetric(MIN_VALUE)

  def __gt__(self, other):
    """Returns true iff this metric is better than other according to metric

    For MaxMetric, larger values are better
    """
    return self.value > other.value


class MetricFunction(object):
  def __init__(self, metric_fn, metric_type):
    """Metric class which wraps metric computing functions

    Parameters
    ----------
    metric_fn : function
      Function which takes prediction and target and returns the metric value
    metric_type : int
      Either `MIN_METRIC`, which indicates that for this metric, lower values
      are better, or `MAX_METRIC`, which indicates that for this metric,
      higher values are better
    """
    self.metric_fn = metric_fn
    self.metric_type = metric_type

  def __call__(self, prediction, target):
    """Computes value of metric between prediction and target

    Parameters
    ----------
    prediction : torch.autograd.Variable
      Predicted value
    target : torch.autograd.Variable
      Target value
    """
    value = self.metric_fn(prediction, target)
    return self.metric_type(value)


METRICS = {
    'psnr': MetricFunction(image_metrics.compute_psnr, MaxMetric),
    'ssim': MetricFunction(image_metrics.compute_ssim, MaxMetric),
    'binary_accuracy': MetricFunction(scalar_metrics.binary_accuracy,
                                      MaxMetric)
}


def get_metric_fn(metric_name):
  assert metric_name in METRICS, 'Unknown metric {}'.format(metric_name)
  return METRICS[metric_name]


def get_loss_metric(value):
  return MinMetric(value)


def accumulate_metric(dictionary, metric_name, metric):
  if metric_name in dictionary:
    dictionary[metric_name].accumulate(metric)
  else:
    dictionary[metric_name] = metric

