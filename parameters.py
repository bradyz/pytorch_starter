import argparse

import torch


DATASET_NAMES = ['CIFAR10']


def _bool(maybe_raw_string):
    if maybe_raw_string in [True, False]:
        return maybe_raw_string

    raw_string = str(maybe_raw_string).lower()

    if raw_string in ['0', 'false']:
        return False
    elif raw_string in ['1', 'true']:
        return True

    raise ValueError('%s is not castable to bool.' % maybe_raw_string)


def _choice(choices):
    def func(raw_string):
        if raw_string in choices:
            return raw_string

        raise ValueError('%s is not in %s.' % (raw_string, choices))

    return func


class Parameters(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.keyword_order = list()
        self.keyword_cast_func = dict()

    def add_keyword(self, key, key_type, default_val=None):
        if key not in self.keyword_order:
            self.keyword_order.append(key)

        self.__dict__[key] = default_val
        self.keyword_cast_func[key] = key_type

    def parse(self):
        for key in self.keyword_order:
            val = self.__dict__[key]

            self.parser.add_argument(
                    '--%s' % key, required=val is None, default=val)

        for key, val in vars(self.parser.parse_args()).items():
            self.__dict__[key] = self.keyword_cast_func[key](val)

    def state_dict(self):
        return {key: self.__dict__[key] for key in self.keyword_order}

    def load_state_dict(self, state):
        for key, val in state.items():
            self.__dict__[key] = val

    def __str__(self):
        result = list()

        for key in self.keyword_order:
            val = self.__dict__[key]

            result.append('%s (%s): %s' % (key, val.__class__.__name__, val))

        return '\n'.join(result)


class DefaultParameters(Parameters):
    def __init__(self):
        super().__init__()

        self.add_keyword('checkpoint_path', str)
        self.add_keyword('data_dir', str)
        self.add_keyword('dataset_name', _choice(DATASET_NAMES))

        self.add_keyword('batch_size', int, 64)
        self.add_keyword('lr', float, 1e-4)
        self.add_keyword('min_lr_ratio', float, 1e-2)
        self.add_keyword('max_epoch', int, 200)

        self.add_keyword('weight_decay', float, 5e-4)

        self.add_keyword('use_visdom', _bool, True)
        self.add_keyword('use_tqdm', _bool, True)
        self.add_keyword('use_cuda', _bool, torch.cuda.is_available())
