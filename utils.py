import torch


def make_variable(tensor, use_cuda, requires_grad=False):
    if use_cuda:
        tensor = tensor.cuda()

    return torch.autograd.Variable(tensor, requires_grad=requires_grad)


def modify_learning_rate(epoch, optimizer, params):
    """
    Decays linearly from params.lr to params.lr * params.min_lr_ratio.

    For example - 1e-2 to 1e-4 if ratio is 0.01.
    """
    alpha = (epoch - 1) / params.max_epoch

    lr = (1.0 - alpha) * params.lr + alpha * params.lr * params.min_lr_ratio

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save(state, checkpoint_path):
    """
    Every value in state must overload state_dict().
    """
    torch.save(
            {key: value.state_dict() for key, value in state.items()},
            checkpoint_path)


def maybe_load(state, checkpoint_path):
    """
    Every value in state must overload load_state_dict().
    """
    print('Loading %s.' % checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path)
    except FileNotFoundError:
        print('Failed. Starting from scratch.')

        return False

    for key, value in state.items():
        try:
            value.load_state_dict(checkpoint[key])
        except Exception as e:
            print(e)
            print('Failed. Could not load %s' % key)

            return False

    print('Success.')
