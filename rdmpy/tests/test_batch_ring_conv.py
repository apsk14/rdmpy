import unittest
from ..blur import ring_convolve, batch_ring_convolve
import torch


class TestRingConvolveFunctions(unittest.TestCase):
    def test_batch_ring_convolve(self):
        """Test that batched and normal ring convolution are numerically close"""
        img_stack = torch.rand((3, 3, 256, 256))
        psfs = torch.rand((256, 513, 256))

        batched = batch_ring_convolve(img_stack, psfs, device="cpu")
        unbatched = torch.zeros_like(img_stack)
        for i in range(3):
            for j in range(3):
                unbatched[i, j] = ring_convolve(img_stack[i, j], psfs, device="cpu")

        self.assertTrue(
            torch.all(torch.isclose(batched, unbatched, rtol=0.0001)).item()
        )


if __name__ == "__main__":
    unittest.main()
