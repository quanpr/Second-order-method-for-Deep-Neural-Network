# Second-order method for Deep Neural Network


Codes for ECE 236C to fullfill the course requriements.

To reproduce the results in the report, for Conjugate Graident (CG) solver, please run:

```
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dir DIR --epoch 300 --lr_decay 300 --cuda --optim NewtonACR_CG --freq 1 --init 0.005 --step 1 --model CNN --N1 100 --bs 512 --rho 1 --mGPUs
```

For Gradient Descent (GD) solver of Trust Region problem, please run:

```
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dir DIR --epoch 300 --lr_decay 30 --cuda --optim Newton --freq 1 --init 0.05 --step 0.05 --bs 512 --decay 0 --momentum 0 --lr 0.1
```

OR 

```
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dir DIR --epoch 300 --lr_decay 30 --cuda --optim NewtonCR --freq 1 --init 0.05 --step 0.05 --bs 512 --decay 0 --momentum 0 --lr 0.1
```

for Cubic Regularization problem.

Should you have any questions, please kindly contact me at <prquan@g.ucla.edu>
