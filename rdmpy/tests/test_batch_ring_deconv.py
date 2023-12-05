import logging
import numpy as np
import time
import torch
import unittest

from ..deblur import ring_deconvolve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TestRingConvolveFunctions(unittest.TestCase):
    def test_batch_ring_deconvolve(self):
        """Test that lri deconvolution with batched and normal ring convolution are numerically close"""
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"

        img_stack = torch.rand((384, 384)).to(device)
        psfs = torch.rand((384, 769, 384)).to(device)

        t1 = time.time()
        using_unbatched_conv = ring_deconvolve(
            img_stack, psfs, use_batch_conv=False, device=device
        )
        t2 = time.time()
        logging.info(
            f"Unbatched-conv deconvolution time: {t2 - t1:2f} seconds per image on {device}"
        )

        t1 = time.time()
        using_batched_conv = ring_deconvolve(
            img_stack, psfs, use_batch_conv=True, device=device
        )
        t2 = time.time()
        logging.info(
            f"Batched-conv deconvolution time: {t2 - t1:2f} seconds per image on {device}"
        )

        self.assertTrue(
            np.max(np.abs(using_batched_conv - using_unbatched_conv)) < 0.0003
        )


if __name__ == "__main__":
    unittest.main()
