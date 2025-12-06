
from abc import ABC, abstractmethod
# ABC is the base class for defining Abstract Base Classes in Python.
# By inheriting from ABC, you can define a class that contains abstract methods (decorated with @abstractmethod),
# which must be implemented in any derived subclass.

import numpy as np
import torch as th
import torch.distributed as dist

def create_named_schedule_sampler(name, diffusion):
    """
    Factory function that creates and returns a ScheduleSampler instance based on the provided name.
    This function acts as a selector for different types of pre-defined samplers depending on user input.

    Args:
        name (str): The name of the sampler to be created.
        diffusion: The diffusion object used for sampling.

    Returns:
        An instance of the corresponding ScheduleSampler.

    Raises:
        NotImplementedError: If the sampler name is not recognized.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"Unknown schedule sampler: {name}")

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    LossAwareSampler is a "loss-aware" schedule sampler.
    It adjusts the sampling weights based on the loss values at different timesteps,
    so the model pays more attention to timesteps with higher loss.
    This approach can help accelerate convergence during training.
    """
    
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using loss values from the model.
        This method should be called on each process (rank) with a batch of
        sampled timesteps and their corresponding losses.
        It performs synchronization to ensure that all ranks maintain exactly
        the same reweighting scheme.

        Args:
            local_ts (Tensor): A 1D tensor of integer timesteps.
            local_losses (Tensor): A 1D tensor of loss values corresponding to each timestep.
        """
        # Gather the batch sizes (i.e., lengths of local_ts) from all processes
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad each gathered batch to the maximum batch size
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for _ in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for _ in batch_sizes]

        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # Extract timesteps and losses, trimming the padding
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]

        self.update_with_all_losses(timesteps, losses)
        
    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.
        Sub-classes should override this method to update the reweighting using losses from the model.
        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.
        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """



class LossSecondMomentResampler(LossAwareSampler):
    """
    LossSecondMomentResampler is a sampler that inherits from LossAwareSampler.
    Its primary function is to adjust sampling weights based on the second moment
    (i.e., the mean of the squared loss values) of the loss for each timestep.

    By focusing more on timesteps with higher loss variance, this strategy encourages the model to learn more effectively from harder examples and can speed up convergence.
    """
    
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        """
        Args:
            diffusion: The diffusion object that provides the number of timesteps.
            history_per_term (int): Number of recent loss values to retain per timestep.
                                    Default is 10, meaning the last 10 loss values will be stored.
            uniform_prob (float): A small probability to assign uniform weights during sampling
                                  to maintain a degree of exploration. Default is 0.001.
        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """
        Compute the sampling weights based on the second moment of historical losses.

        Returns:
            A numpy array of sampling weights for each timestep.
        """
        if not self._warmed_up():
            # If warm-up is not complete, use uniform weights
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)

        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)  # Normalize
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)  # Add uniform probability for stability
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        Update the loss history with a new batch of losses.

        Args:
            ts (list[int]): List of timestep indices.
            losses (list[float]): List of corresponding loss values.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss entry and insert the new one
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        Check if loss history has been fully populated for all timesteps.

        Returns:
            bool: True if every timestep has `history_per_term` losses recorded, False otherwise.
        """
        return (self._loss_counts == self.history_per_term).all()