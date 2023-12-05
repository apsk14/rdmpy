# Testing

Testing suites using python unittest. Currently implemented:

- [batched convolution numerical & speed comparison](test_batch_ring_conv.py)
- [batched deconvolution numerical & speed comparison](test_batch_ring_deconv.py)

To run all tests, run the following command from the root project directory:

```sh
python -m unittest rdmpy/tests/*.py
```
NOTE: timing here uses *random PSF tensors*, which are harder to optimize than real, sparse PSF data. Speedup with real PSF data are likely to be orders of magnitude greater.

Current outputs:
```
2023-12-04 19:41:35,909 - INFO - Unbatched convolution deconvolution time: 0.061296 seconds per image on cuda:0
2023-12-04 19:41:37,564 - INFO - Batched convolution deconvolution time: 0.016551 seconds per image on cuda:0
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:40<00:00,  7.43it/s]
2023-12-04 19:42:22,317 - INFO - Unbatched-conv deconvolution time: 40.483006 seconds per image on cuda:0
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:03<00:00, 94.58it/s]
2023-12-04 19:41:41,834 - INFO - Batched-conv deconvolution time: 3.268919 seconds per image on cuda:0
.
----------------------------------------------------------------------
Ran 2 tests in 56.421s

OK
```