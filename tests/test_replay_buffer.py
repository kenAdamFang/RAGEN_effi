import unittest
import collections # For deque if needed in tests, ReplayBuffer uses it internally
import random # For setting seed if needed for reproducible sampling tests (optional)

# Attempt to import torch, skip torch-dependent tests if unavailable
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ragen.trainer.replay_buffer import ReplayBuffer
from verl import DataProto # Assuming this path is correct and verl is installed

# Helper to create dummy DataProto objects for testing
def create_dummy_dp(idx: int) -> DataProto:
    # Create a DataProto with some minimal, distinguishable data
    if TORCH_AVAILABLE:
        batch_content = {'id_tensor': torch.tensor([idx])}
    else:
        batch_content = {'id_list': [idx]} # Fallback if torch is not available

    return DataProto(
        batch=batch_content,
        non_tensor_batch={'id_val': idx},
        meta_info={'id': idx, 'desc': f'dp_{idx}'}
    )

class TestReplayBuffer(unittest.TestCase):
    def test_initialization(self):
        buffer = ReplayBuffer(capacity=10)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.capacity, 10)
        # ReplayBuffer.__init__ from step 2 was (self, capacity, sampling_batch_size)
        # The integration in agent_trainer.py used ReplayBuffer(capacity=...)
        # This test assumes the constructor is ReplayBuffer(self, capacity) or that sampling_batch_size has a default
        # If ReplayBuffer indeed stores sampling_batch_size, that should be tested too.
        # For now, assuming it's not stored or used directly by ReplayBuffer's core logic beyond what sample() receives.

    def test_add_single_experience(self):
        buffer = ReplayBuffer(capacity=5)
        dp1 = create_dummy_dp(1)
        buffer.add(dp1)
        self.assertEqual(len(buffer), 1)
        # Check if the item is actually in (by sampling)
        sampled = buffer.sample(current_batch_size=1)
        self.assertEqual(len(sampled), 1)
        self.assertEqual(sampled[0].non_tensor_batch['id_val'], 1)

    def test_len_updates_correctly(self):
        buffer = ReplayBuffer(capacity=5)
        self.assertEqual(len(buffer), 0)
        buffer.add(create_dummy_dp(1))
        self.assertEqual(len(buffer), 1)
        buffer.add(create_dummy_dp(2))
        self.assertEqual(len(buffer), 2)

    def test_buffer_exceeds_capacity_fifo(self):
        capacity = 3
        buffer = ReplayBuffer(capacity=capacity)
        
        dps = [create_dummy_dp(i) for i in range(capacity + 2)] # dp_0, dp_1, dp_2, dp_3, dp_4

        # Fill the buffer
        for i in range(capacity): # Add dp_0, dp_1, dp_2
            buffer.add(dps[i])
        self.assertEqual(len(buffer), capacity)

        # Add dp_3, which should evict dp_0
        buffer.add(dps[capacity]) # dps[3] is dp_3
        self.assertEqual(len(buffer), capacity)
        
        # Sample all items to check contents
        sampled_all = buffer.sample(current_batch_size=capacity)
        ids_in_buffer = sorted([dp.non_tensor_batch['id_val'] for dp in sampled_all])
        
        self.assertNotIn(dps[0].non_tensor_batch['id_val'], ids_in_buffer, "Oldest item (dp_0) should have been evicted.")
        self.assertIn(dps[capacity].non_tensor_batch['id_val'], ids_in_buffer, "New item (dp_3) should be present.")
        expected_ids_after_dp3 = sorted([dp.non_tensor_batch['id_val'] for dp in dps[1:capacity+1]]) # dp_1, dp_2, dp_3
        self.assertListEqual(ids_in_buffer, expected_ids_after_dp3)

        # Add dp_4, which should evict dp_1
        buffer.add(dps[capacity + 1]) # dps[4] is dp_4
        self.assertEqual(len(buffer), capacity)

        sampled_all_again = buffer.sample(current_batch_size=capacity)
        ids_in_buffer_again = sorted([dp.non_tensor_batch['id_val'] for dp in sampled_all_again])

        self.assertNotIn(dps[1].non_tensor_batch['id_val'], ids_in_buffer_again, "Item dp_1 should have been evicted.")
        self.assertIn(dps[capacity + 1].non_tensor_batch['id_val'], ids_in_buffer_again, "New item (dp_4) should be present.")
        expected_ids_after_dp4 = sorted([dp.non_tensor_batch['id_val'] for dp in dps[2:capacity+2]]) # dp_2, dp_3, dp_4
        self.assertListEqual(ids_in_buffer_again, expected_ids_after_dp4)

    def test_sample_from_empty_buffer(self):
        buffer = ReplayBuffer(capacity=5)
        samples = buffer.sample(current_batch_size=3)
        self.assertEqual(len(samples), 0)
        self.assertIsInstance(samples, list)

    def test_sample_less_than_batch_size(self):
        buffer = ReplayBuffer(capacity=5)
        dp1 = create_dummy_dp(1)
        dp2 = create_dummy_dp(2)
        buffer.add(dp1)
        buffer.add(dp2)
        
        samples = buffer.sample(current_batch_size=3)
        self.assertEqual(len(samples), 2)
        # Verify the content of sampled items
        sampled_ids = sorted([s.non_tensor_batch['id_val'] for s in samples])
        self.assertEqual(sampled_ids, [1, 2])
        for item in samples:
            self.assertIsInstance(item, DataProto)

    def test_sample_more_than_buffer_size_equals_sample_all(self):
        buffer = ReplayBuffer(capacity=5)
        dps_to_add = [create_dummy_dp(i) for i in range(3)] # Add 3 items: dp_0, dp_1, dp_2
        for dp in dps_to_add:
            buffer.add(dp)
        
        # Try to sample 5 (more than 3 available)
        samples = buffer.sample(current_batch_size=5)
        self.assertEqual(len(samples), 3)
        sampled_ids = sorted([s.non_tensor_batch['id_val'] for s in samples])
        self.assertEqual(sampled_ids, [0, 1, 2])
        for item in samples:
            self.assertIsInstance(item, DataProto)

    def test_sample_exact_batch_size_available(self):
        buffer = ReplayBuffer(capacity=5)
        dps_to_add = [create_dummy_dp(i) for i in range(3)] # dp_0, dp_1, dp_2
        for dp in dps_to_add:
            buffer.add(dp)
        
        samples = buffer.sample(current_batch_size=3)
        self.assertEqual(len(samples), 3)
        sampled_ids = sorted([s.non_tensor_batch['id_val'] for s in samples])
        self.assertEqual(sampled_ids, [0, 1, 2]) # Should contain all added items
        for item in samples:
            self.assertIsInstance(item, DataProto)

    def test_sample_batch_size_from_larger_buffer(self):
        buffer = ReplayBuffer(capacity=10)
        dps_to_add = [create_dummy_dp(i) for i in range(10)] # dp_0 to dp_9
        for dp in dps_to_add:
            buffer.add(dp)
        
        sample_size = 5
        samples = buffer.sample(current_batch_size=sample_size)
        self.assertEqual(len(samples), sample_size)
        
        # Ensure all sampled items are unique and from the buffer
        sampled_ids = [s.non_tensor_batch['id_val'] for s in samples]
        self.assertEqual(len(set(sampled_ids)), sample_size, "Sampled items should be unique.")
        
        original_ids = [dp.non_tensor_batch['id_val'] for dp in dps_to_add]
        for sid in sampled_ids:
            self.assertIn(sid, original_ids, "Sampled item must be one of the items added.")
            
        for item in samples:
            self.assertIsInstance(item, DataProto)

    def test_sampling_is_random(self):
        # This test is statistical in nature and might occasionally fail
        # even if the implementation is correct.
        # A simpler check is that two consecutive samples of the same size
        # from a larger buffer are not always identical.
        capacity = 20
        buffer = ReplayBuffer(capacity=capacity)
        for i in range(capacity):
            buffer.add(create_dummy_dp(i))

        sample_size = 5
        
        # For reproducibility of this specific test run if desired
        # random.seed(42) 
        
        samples1_ids = sorted([s.non_tensor_batch['id_val'] for s in buffer.sample(sample_size)])
        samples2_ids = sorted([s.non_tensor_batch['id_val'] for s in buffer.sample(sample_size)])

        # It's highly probable they are different if buffer_size >> sample_size
        # If capacity is large and sample_size is small, it's very unlikely to get the same set.
        # If they are the same, it doesn't necessarily mean it's not random, but it's less likely.
        # We run this a few times. If all are same, then it's suspicious.
        
        attempts = 5
        all_samples_identical = True
        previous_sample_ids = sorted([s.non_tensor_batch['id_val'] for s in buffer.sample(sample_size)])

        for _ in range(attempts -1):
            current_sample_ids = sorted([s.non_tensor_batch['id_val'] for s in buffer.sample(sample_size)])
            if current_sample_ids != previous_sample_ids:
                all_samples_identical = False
                break
            previous_sample_ids = current_sample_ids
        
        if len(buffer) > sample_size : # Only assert if there's actual room for randomness
             self.assertFalse(all_samples_identical, 
                             f"Sampling {attempts} times of size {sample_size} from buffer of size {len(buffer)} "
                             f"yielded identical sets. This is unlikely if sampling is random. First sample: {samples1_ids}")


if __name__ == '__main__':
    # The create_dummy_dp helper uses torch.tensor if TORCH_AVAILABLE is True.
    # The tests will run, and if torch is not installed, the dummy DataProto
    # will use a list for its 'batch' attribute, which is fine for testing
    # the replay buffer's mechanics.
    unittest.main()
