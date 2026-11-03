# References and implementation attribution

This repository's Python training and model code was written for this task; it is not a
copy of an external implementation. The design uses the following published methods:

- Zagoruyko, S. and Komodakis, N. (2016). **Wide Residual Networks.**
  https://arxiv.org/abs/1605.07146 — widened residual blocks and network design.
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). **Identity Mappings in Deep Residual Networks.**
  https://arxiv.org/abs/1603.05027 — pre-activation residual design.
- Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D. (2018).
  **mixup: Beyond Empirical Risk Minimization.** https://arxiv.org/abs/1710.09412
- Yun, S. et al. (2019). **CutMix: Regularization Strategy to Train Strong Classifiers
  with Localizable Features.** https://arxiv.org/abs/1905.04899
- Cubuk, E. D., Zoph, B., Shlens, J., and Le, Q. V. (2020).
  **RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.**
  https://arxiv.org/abs/1909.13719 — implemented by torchvision.transforms.RandAugment.
- Loshchilov, I. and Hutter, F. (2017). **SGDR: Stochastic Gradient Descent with Warm Restarts.**
  https://arxiv.org/abs/1608.03983 — cosine annealing; this repo uses no restarts.
- PyTorch automatic mixed precision documentation:
  https://docs.pytorch.org/docs/stable/amp.html
- Torchvision RandAugment documentation:
  https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html

Additional standard techniques: exponential moving averages of weights, random crop/flip,
per-channel training-set normalization, Nesterov SGD, and horizontal-flip prediction averaging.
No reference repository's weights, dataset downloads, or external training examples are used.

## Wider / AugMix experiments

- Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., and Lakshminarayanan, B.
  (2020). **AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty.**
  https://arxiv.org/abs/1912.02781 — augmentation mixtures and three-view consistency training.
- Torchvision AugMix implementation and API:
  https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.AugMix.html

The custom validation stress test is implemented locally using PIL/NumPy transformations
of held-out allowed training images. It does not download or use CIFAR-10-C or any other
external evaluation dataset, and is not claimed as a standardized corruption benchmark.

## Optional test-time BN recalibration

PyTorch BatchNorm2d documentation (running statistics, evaluation mode, cumulative-average
momentum): https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
The implementation is one-pass BN buffer recalibration, not gradient-based test-time
optimization. Test use is explicitly gated on organizer permission.
