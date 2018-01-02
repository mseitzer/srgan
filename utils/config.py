import json
import os

JSON_TYPE_TAG = '__type__'


def _decode_config(root_path):
  def _decode(obj):
    res = {}
    if '#include' in obj:
      includes = obj['#include']
      if not isinstance(includes, list):
        includes = [includes]
      for path in includes:
        if not os.path.isabs(path):
          path = os.path.join(os.path.dirname(root_path), path)

        sub_conf = Configuration.from_json(path)
        res.update(sub_conf.__dict__)
      del obj['#include']

    if JSON_TYPE_TAG in obj and str(Configuration) == obj[JSON_TYPE_TAG]:
      res.update(obj)
      return Configuration.from_dict(res)
    else:
      res.update(obj)
      return res

  return _decode


class Configuration(object):
  """Configuration object which offers convenience methods for serializing
  and deserializing
  """
  def __init__(self):
    self.seed = 0
    self.__dict__[JSON_TYPE_TAG] = str(type(self))
    super(Configuration, self).__init__()

  def __str__(self):
    """Pretty stringify configuration"""
    s = 'Configuration object\n'
    for key, value in self.__dict__.items():
      s += '  {}: {}\n'.format(key, value)
    return s

  @property
  def file(self):
    return self._src_file

  def has_attr(self, key):
    """Checks if configuration has an attribute

    Parameters
    ----------
    key : string
      Name of attribute
    """
    return hasattr(self, key)

  def get_attr(self, key, default=None, alternative=None):
    """Returns attribute of configuration or default value

    Parameters
    ----------
    key : string
      Name of attribute
    default : object
      Value to return if attribute does not exist
    alternative : string
      Key of alternative attribute to use if configuration does not have
      requested key. Raises an error if the alternative key does also not exist
    """
    if hasattr(self, key):
      return getattr(self, key)
    else:
      value = default
      if alternative is not None:
        value = self.get_attr(alternative)
        if value is None:
          raise ValueError(('Configuration did not contain {} '
                           ' or alternative {}').format(key, alternative))
      return value

  def serialize(self, dst):
    """Serialize configuration to JSON file

    Parameters
    ----------
    dst : string
      Destination file path
    """
    with open(dst, 'w') as f:
      json.dump(self.__dict__, f,
                default=lambda obj: obj.__dict__,
                indent=2)

  def update(self, values_by_keys):
    """Adds values to this configuration

    Attempts to convert string values into python primitive types. Currently
    supported are the types bool, int, float, list.
    If no conversion succeeds, uses the value as string.

    Parameters
    ----------
    values_by_keys : dict of string -> string
      Dictionary which contains key: value pairs to update configuration from
    """
    def convert(s):
      if (s.startswith('[') and s.endswith(']')) or \
         (s.startswith('(') and s.endswith(')')):
         # This, of course, breaks badly for nested lists
         return [convert(elem.strip()) for elem in s[1:-1].split(',')]

      if s == 'False':
        return False
      elif s == 'True':
        return True

      try:
        return int(s)
      except ValueError:
        pass

      try:
        return float(s)
      except ValueError:
        pass

      return s

    for key, value in values_by_keys.items():
      self.__dict__[key] = convert(value)

  def to_param_dict(self, required_params=[], optional_params=[],
                    key_renames={}):
    """Converts configuration to a dict which can be passed to function call

    Parameters
    ----------
    required_params : list of string
      List of attribute names the configuration is required to have for the
      conversion to work
    optional_params : list of string or dict of string -> object
      If list, contains the keys of optional configuration attributes to be
      inserted in the result. If dict, additionally specifies default values
      to be used if the configuration does not contain an optional attribute
    key_renames : dict of string -> string
      Dictionary which maps configuration keys to keys in the output dictionary
    """
    params = {}
    for key in required_params:
      value = self.get_attr(key)
      assert value is not None, \
          'Parameter {} is marked as required'.format(key)
      params[key] = value

    if isinstance(optional_params, dict):
      for key, default_value in optional_params.items():
        value = self.get_attr(key, default=default_value)
        params[key] = value
    else:
      for key in optional_params:
        value = self.get_attr(key)
        if value is not None:
          params[key] = value

    return {key_renames.get(key, key): value for key, value in params.items()}

  @staticmethod
  def from_dict(dictionary):
    """Construct configuration from dictionary

    Parameters
    ----------
    dictionary : dict
      Dictionary to convert

    Returns : Configuration
    -------
      Converted configuration object
    """
    if isinstance(dictionary, Configuration):
      return dictionary
    conf = Configuration()
    conf.__dict__.update(dictionary)
    return conf

  @staticmethod
  def from_json(src):
    """Deserialize configuration from JSON file

    Parameters
    ----------
    src : string
      Source file path

    Returns : Configuration
    -------
      Deserialized configuration object
    """
    with open(src, 'r') as f:
      conf = json.load(f, object_hook=_decode_config(src))

    if isinstance(conf, dict):
      conf = Configuration.from_dict(conf)

    conf._src_file = src

    if hasattr(conf, 'include'):
      for key, path in conf.include.items():
        if not os.path.isabs(path):
          path = os.path.join(os.path.dirname(src), path)

        sub_conf = Configuration.from_json(path)

        if key == '':
          conf.__dict__ = dict(**sub_conf.__dict__, **conf.__dict__)
        else:
          saved_value = conf.get_attr(key, default=None)
          conf.__dict__[key] = sub_conf.__dict__
          if (isinstance(conf.__dict__[key], dict) and
              isinstance(saved_value, dict)):
            conf.__dict__[key].update(saved_value)
      del conf.__dict__['include']

    return conf
