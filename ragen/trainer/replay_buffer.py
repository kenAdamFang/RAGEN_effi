import collections
import random
from verl import DataProto

class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experiences.
    """
    def __init__(self, capacity: int, sampling_batch_size: int = None):
        """
        Initialize the ReplayBuffer.

        Args:
            capacity: The maximum number of experiences to store in the buffer.
            sampling_batch_size: (Optional) This parameter is noted but not used internally
                                 for sampling logic, as batch size is provided to `sample()`.
        """
        self.capacity = capacity
        # self.sampling_batch_size = sampling_batch_size # Removed as per requirement
        self._buffer = collections.deque(maxlen=capacity)

    def add(self, experience: DataProto):
        """
        Add an experience to the buffer.

        Args:
            experience: The DataProto object representing the experience.
        """
        self._buffer.append(experience)

    def sample(self, current_batch_size: int = None) -> list[DataProto]:
        """
        Sample a batch of experiences from the buffer.

        Args:
            current_batch_size: The number of experiences to sample.
                                If the buffer has fewer experiences than current_batch_size,
                                all experiences are returned.

        Returns:
            A list of DataProto objects.
        """
        if current_batch_size is None:
            raise ValueError("current_batch_size must be provided to sample method.")

        num_experiences = len(self._buffer)
        actual_sample_size = min(current_batch_size, num_experiences)

        if actual_sample_size == 0:
            return []

        return random.sample(list(self._buffer), actual_sample_size)

    def __len__(self) -> int:
        """
        Return the current number of experiences in the buffer.
        """
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, current_size={len(self._buffer)})"
