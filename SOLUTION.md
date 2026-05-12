# SOLUTION

```bash
pip install -r requirements.txt
python validate.py --data_dir ./data --batch_size 256 --n_batches 32 --output results.json
```

### Modified files

**`head_init.py`**
- Ridge / least-squares linear classifier fitted on top of frozen ResNet-18 features extracted from the full CIFAR-100 training set (50 000 samples).
- Backbone `fc` is replaced with `nn.Identity()` to extract 512-dimensional features.
- One-hot targets are constructed and the normal equations are solved with `torch.linalg.solve`: `W = (X^T X + alpha N I)^{-1} X^T Y`, `alpha = 0.01`.
- Bias is set to the mean residual: `bias = mean(Y - X W)`.

**`zo_optimizer.py`**
- `_CURRICULUM` is set to tune only `layer4.1.bn2.weight` and `layer4.1.bn2.bias` for the entire run. The `fc` layer is kept frozen because the ridge-fitted head is already strong and SPSA updates in the 51 200-dimensional `fc` space add noise rather than signal.
- SPSA-style antithetic estimator: for each step, a per-parameter unit-normalised Gaussian direction `u` is sampled; `f_plus` and `f_minus` are evaluated with `±eps * u`; pseudo-gradient is `(f_plus - f_minus) / (2 * eps) * u`. Costs exactly 2 forward passes per step.
- `f_plus` from the first sample is reused as `loss_before`, saving one extra forward pass.
- Adam update (beta1=0.9, beta2=0.999, eps=1e-8) with global gradient norm clipping (`max_grad_norm=1.0`).
- LR schedule: linear warm-up for the first 10% of steps, then cosine decay.

**`augmentation.py`**
- Added `T.RandomCrop(224, padding=28)`, `T.ColorJitter(0.3, 0.3, 0.3, 0.05)`, `T.RandomGrayscale(p=0.1)`, `T.RandomErasing(p=0.2, scale=(0.02, 0.2))` to the training pipeline.

**`train_data.py`**
- Added `drop_last=True` to the training `DataLoader` for consistent batch sizes.

### Key findings

The dominant factor was the quality of the head initialisation. The pretrained ResNet-18 backbone already provides useful 512-dimensional features; solving the final linear classification problem on those frozen features with ridge regression was far more reliable than trying to learn the head through noisy zero-order updates.

Once the `fc` layer is well-initialised, allowing ZO to perturb it degrades accuracy because random perturbations in a 51 200-dimensional space are almost orthogonal to any useful descent direction. Restricting ZO to the 1 024-parameter BatchNorm affine pair `layer4.1.bn2` was the only consistently stable backbone update.
