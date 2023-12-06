import logging
import numpy as np
import time
import torch
import unittest

from ..blur import ring_convolve, batch_ring_convolve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TestRingConvolveFunctions(unittest.TestCase):
    def test_batch_ring_convolve(self):
        """Test that batched and normal ring convolution are numerically close"""
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"

        img_stack = torch.rand((10, 10, 384, 384)).to(device)
        psfs = torch.rand((384, 769, 384)).to(device)

        t1 = time.time()
        unbatched = torch.zeros_like(img_stack)
        for i in range(img_stack.shape[0]):
            for j in range(img_stack.shape[1]):
                unbatched[i, j] = ring_convolve(
                    img_stack[i, j], psfs, device=device, verbose=False
                )
        t2 = time.time()
        logging.info(
            f"Unbatched convolution deconvolution time: {(t2 - t1)/100:2f} seconds per image on {device}"
        )

        t1 = time.time()
        batched = batch_ring_convolve(img_stack, psfs, device=device)
        t2 = time.time()
        logging.info(
            f"Batched convolution deconvolution time: {(t2 - t1)/100:2f} seconds per image on {device}"
        )

        self.assertTrue(torch.allclose(batched, unbatched, 0.0001))


if __name__ == "__main__":
    unittest.main()
