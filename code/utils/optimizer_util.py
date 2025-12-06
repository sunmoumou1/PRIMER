def optim_warmup(step, optim, lr, warmup_iters):
    """
    Implements learning rate warm-up.

    During the initial phase of training, the learning rate is gradually increased
    from zero to the target learning rate. This helps prevent unstable updates
    caused by using a high learning rate at the very beginning.

    Args:
        step (int): The current training step.
        optim (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
        lr (float): The target learning rate after warm-up.
        warmup_iters (int): Number of warm-up iterations.
    """
    lr = lr * float(step) / warmup_iters
    for param_group in optim.param_groups:
        param_group["lr"] = lr


def update_ema(model, ema_model, ema_rate):
    '''
        p2 = ema_rate * p2 + (1 - ema_rate) * p1
    '''
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        # Beta * previous ema weights + (1 - Beta) * current non ema weight
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def optim_decay(optim, minimum_lr=None):
    '''
        optim_decay function implements a learning rate decay strategy. 
    '''

    # Update the learning rate for each parameter group in the optimizer
    for param_group in optim.param_groups:
        # Get the current learning rate from the optimizer
        current_lr = param_group['lr']
        # Update the learning rate based on the decay factor
        updated_lr = current_lr * 0.5
        
        if minimum_lr is not None:
            if updated_lr < minimum_lr:
                updated_lr = minimum_lr
                
        param_group['lr'] = updated_lr
