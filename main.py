import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
# from torchvision.models import resnet18,resnet34,resnet50,vgg16,inception_v3
from models import resnet18,resnet34,resnet50, VGG
import numpy as np
import torch.utils.data as td
import random,time
from cnn import CNN, CNN_ReLU
from mlp import MLP
import argparse
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
from optim import trust_region_newton_step, adaptive_trust_region_newton_step, trust_region_newton_step2
from newton_cg import trust_region_newton_cg
from newton_cr import newton_step_cubic_regularization, newton_step_cr_cg, newton_adaptive_cubic_regularization, \
						newton_adaptive_cr_cg
import random

# Then set a random seed for deterministic results/reproducability.
SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def parse_args():
	parser = argparse.ArgumentParser(description='Train a classifier for CIFAR10')
	parser.add_argument('--model', dest='model',
					  help='classifier type',
					  default='CNN', type=str)
	parser.add_argument('--lr', dest='lr',
					  help='learning rate',
					  default=0.1, type=float)
	parser.add_argument('--momentum', dest='momentum',
					  help='momentum of learning',
					  default=0.09, type=float)
	parser.add_argument('--epoch', dest='epoch',
					  help='number of training epoch',
					  default=50, type=int)
	parser.add_argument('--lr_decay', dest='lr_decay',
					  help='learning rate decay',
					  default=20, type=int)
	parser.add_argument('--freq', dest='print_frq',
					  help='log printing frequency',
					  default=50, type=int)
	parser.add_argument('--resume', '-r', action='store_true',
					  help='resume from checkpoint')
	parser.add_argument('--checkpoint', dest='checkpoint',
					  help='ith checkpoint to resume',
					  default=5, type=int)
	parser.add_argument('--mGPUs', dest='mGPUs',
					  help='whether use multiple GPUs',
					  action='store_true')
	parser.add_argument('--cuda', dest='cuda',
						help='whether use CUDA',
						action='store_true')
	parser.add_argument('--dir', dest='dir',
					  help='model saving directory',
					  default='model', type=str)
	parser.add_argument('--bs', dest='bs',
					  help='batch_size',
					  default=256, type=int)
	parser.add_argument('--optim', '-optim', 
					  help='which optimizer to use',
					  default='SGD', type=str)
	parser.add_argument('--step', dest='step',
					  help='bounded step for trust region',
					  default=0.1, type=float)
	parser.add_argument('--init', dest='init',
					  help='initial step size',
					  default=0.3, type=float)
	parser.add_argument('--tr', dest='tr',
					  help='initial size of trust region',
					  default=0.1, type=float)
	parser.add_argument('--N1', dest='N1',
					  help='iterations prior to reaching boundary',
					  default=10, type=int)
	parser.add_argument('--N2', dest='N2',
					  help='iterations after reaching boundary',
					  default=10, type=int)
	parser.add_argument('--decay', dest='weight_decay',
					  help='weight decay',
					  default=0.0005, type=float)
	parser.add_argument('--eps', dest='eps',
					  help='stopping criterion for conjugate gradient method',
					  default=1e-5, type=float)	
	parser.add_argument('--rho', dest='rho',
					  help='scaling factor for cubic regularization method',
					  default=1e-1, type=float)	
	args = parser.parse_args()
	return args

def adjust_learning_rate(optimizer, epoch, lr_decay=3):
	"""Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
	lr = args.lr * (0.1 ** (epoch // lr_decay))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def cifar_loaders(batch_size, shuffle_test=False): 
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.225, 0.225, 0.225])
	train = datasets.CIFAR10('./', train=True, download=True, 
		transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
		]))
	test = datasets.CIFAR10('./', train=False, 
		transform=transforms.Compose([transforms.ToTensor(), normalize]))
	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
		shuffle=True, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
		shuffle=shuffle_test, pin_memory=True)
	return train_loader, test_loader

args = parse_args()

batch_size = args.bs
test_batch_size = args.bs

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

if args.model == 'CNN':
	model = CNN()
elif args.model == 'CNN_ReLU':
	model = CNN_ReLU()
elif args.model == 'resnet18':
	model = resnet18(num_classes=10)
elif args.model == 'resnet34':
	model = resnet34(num_classes=10)
elif args.model == 'resnet50':
	model = resnet50(num_classes=10)
elif args.model == 'vgg16':
	model = VGG(vgg_name='vgg16', num_classes=10)
elif args.model == 'MLP':
	model = MLP()
else:
	raise ValueError('Unrecognized training model')

if args.optim == 'SGD':
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'LBFGS':
	optimizer = optim.LBFGS(model.parameters(), lr=args.lr)

num_epochs = args.epoch
lr = args.lr
print_itr = args.print_frq
criterion = nn.CrossEntropyLoss()

start_epoch = 0
if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/{}/{}_best.pth'.format(args.dir, args.model))
	model.load_state_dict(checkpoint['net'])
	start_epoch = checkpoint['epoch']

record_loss = []
record_test_loss = []
if args.cuda:
	model.cuda()

if args.mGPUs:
	model = nn.DataParallel(model)

best_accu = 0.0
trust_region = args.tr

print(args)

# implement module for estimating full gradient
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								std=[0.225, 0.225, 0.225])
train = datasets.CIFAR10('./', train=True, download=True, 
		transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
		]))
SAMPLE_RATE = 0.1
full_loader = torch.utils.data.DataLoader(train, batch_size=int(SAMPLE_RATE*(len(train))),
		shuffle=True, pin_memory=True)
full_iterator = iter(full_loader)

for epoch in range(start_epoch, num_epochs):
	model.train()
	total_loss = 0

	start_time = time.time()
	if args.optim == 'SGD':
		lr = args.lr * (0.1 ** (epoch // args.lr_decay))
		adjust_learning_rate(optimizer, epoch, args.lr_decay)
		start = time.time()
		for i, (images, targets) in enumerate(train_loader):

			if args.cuda:
				targets = targets.cuda()
				images = images.cuda()

			model.zero_grad()

			outputs = model(images)
			loss = criterion(outputs, targets)

			loss.backward()
			optimizer.step()
			## TODO
			
			total_loss += loss.item()
			if i % print_itr == 0:
				end = time.time()
				print('Epoch {}: {}/{} | loss {:.4f} | lr {:.6} | time {:.3f}'.format(epoch, i, len(train_loader), loss.item(), lr, end-start))
				start = time.time()
	elif args.optim == 'LBFGS':
		start = time.time()

		def closure():
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, targets)
			# print('loss:', loss.item())
			loss.backward()
			return loss

		for i, (images, targets) in enumerate(train_loader):

			if args.cuda:
				targets = targets.cuda()
				images = images.cuda()

			model.zero_grad()

			# outputs = model(images)
			# loss = criterion(outputs, targets)

			# loss.backward()
			loss = optimizer.step(closure)
			## TODO
			
			total_loss += loss.item()
			if i % print_itr == 0:
				end = time.time()
				print('Epoch {}: {}/{} | loss {:.4f}({:.4f}) | lr {:.6} | time {:.3f}'.format(epoch, i, len(train_loader), loss.item(), total_loss/(i+1), lr, end-start))
				start = time.time()		

	elif args.optim in ('Newton','NewtonCG', 'NewtonCR', 'NewtonACR', 'NewtonACR_CG'):
		start = time.time()
		for i, (images, targets) in enumerate(train_loader):

			if args.cuda:
				targets = targets.cuda()
				images = images.cuda()

			outputs = model(images)
			loss = criterion(outputs, targets)

			if args.optim == 'Newton':
				if epoch != 0 and epoch % args.lr_decay == 0 and i == 0:
					args.step = args.step * 0.1
					args.init = args.init * 0.1
				_, N1 = trust_region_newton_step(args, model, loss)
			elif args.optim == 'NewtonCR':
				if epoch != 0 and epoch % args.lr_decay == 0 and i == 0:
					args.step = args.step * 0.1
					args.init = args.init * 0.1
				_, N1 = newton_step_cubic_regularization(args, model, loss)
				# _, N1 = newton_step_cr_cg(args, model, loss)
			elif args.optim == 'NewtonACR':

				# use a larger batch for full gradient and loss estimation
				try:
					data_full, target_full = next(full_iterator)
				except StopIteration:
					full_iterator =iter(full_loader)
					data_full, target_full = next(full_iterator)
				if args.cuda:
					target_full = target_full.cuda()
					data_full = data_full.cuda()

				if epoch != 0 and epoch % args.lr_decay == 0 and i == 0:
					args.step = args.step * 0.1
					args.init = args.init * 0.1
				loss, N1 = newton_adaptive_cubic_regularization(args, model, images, targets, criterion, loss, data_full, target_full)			
			elif args.optim == 'NewtonACR_CG':

				# use a larger batch for full gradient and loss estimation
				try:
					data_full, target_full = next(full_iterator)
				except StopIteration:
					full_iterator =iter(full_loader)
					data_full, target_full = next(full_iterator)
				if args.cuda:
					target_full = target_full.cuda()
					data_full = data_full.cuda()

				if epoch != 0 and epoch % args.lr_decay == 0 and i == 0:
					args.step = args.step * 0.1
					args.init = args.init * 0.1
				loss, N1, actual_red, pred_red, rho, eps = newton_adaptive_cr_cg(args, model, images, targets, criterion, loss, data_full, target_full)				
			else:
				_, N1 = trust_region_newton_cg(args, model, loss)

			# outputs = model(images)
			# loss = criterion(outputs, targets)			
			# trust_region_newton_step2(args, model, loss, criterion, images, targets)
			# trust_region = adaptive_trust_region_newton_step(args, model, images, \
			# 				targets, trust_region, loss, criterion)

			loss = loss.item() if 'tensor' in str(type(loss)) else loss
			total_loss += loss
			if i % print_itr == 0:
				end = time.time()
				if args.optim in ('NewtonCR', 'NewtonACR'):
					print('Epoch {}: {}/{} | loss {:.4f}({:.4f}) | rho {:7.5f} | step {:.4f} | init {:.4f} | N1 {} | time {:.3f}'.format(epoch, i, len(train_loader), loss, total_loss/(i+1), args.rho, args.step, args.init, N1, end-start))
				elif args.optim == 'NewtonACR_CG':
					print('Epoch {}: {}/{} | loss {:.4f}({:.4f}) | rho {:7.5f} | step {:.4f} | N1 {} | time {:.3f} | actual_red {:7.5e} | pred_red {:7.5e} | ratio {:7.5f} | eps {:.5f}'.format(epoch, i, len(train_loader), loss, total_loss/(i+1), args.rho, args.step, N1, end-start, actual_red, pred_red, rho, eps))
				else:
					print('Epoch {}: {}/{} | loss {:.4f}({:.4f}) | tr {:.4f} | step {:.4f} | init {:.4f} | N1 {} | time {:.3f}'.format(epoch, i, len(train_loader), loss, total_loss/(i+1), args.tr, args.step, args.init, N1, end-start))
				start = time.time()

	else:
		raise ValueError('Unrecognized optimizer')

	total_loss /= len(train_loader)
	end_time = time.time()
	print('In epoch {}, total loss {:.4f}, time {:.2f}s'.format(epoch, total_loss, end_time-start_time))
	record_loss.append(total_loss)
	#Print your results every epoch

	# Evaluate the Model
	print('Finish training epoch {}, now test on test datasets.'.format(epoch))
	correct = 0.
	total = 0.

	model.eval()

	total_test_loss = 0.0
	for i, (images, labels) in enumerate(test_loader):
		
		## Put your prediction code here
		if args.cuda:
			labels = labels.cuda()
			images = images.cuda()

		output_ = model(images)
		loss = criterion(output_, labels)

		values, prediction = torch.max(output_, 1)

		pred = (prediction.long() == labels).sum().float()
		correct += pred
		total += images.shape[0]
		total_test_loss += loss.item()
		# print('{}/{} Accuracy: {}%'.format(i, len(test_loader), 100*pred/images.shape[0]))
	total_test_loss /= len(test_loader)
	record_test_loss.append(total_test_loss)
	accu = correct.float() / total
	print('Accuracy of the model on the test images: {:.3f}%'.format(100 * (accu)))

	print('Saving current model: '+'./checkpoint/{}/{}_current.pth'.format(args.dir, args.model))
	state = {
		'net': model.module.state_dict() if args.mGPUs else model.state_dict(),
		'epoch': epoch,
	}
	if not os.path.isdir('checkpoint/{}/'.format(args.dir)):
		os.mkdir('checkpoint/{}/'.format(args.dir))
	torch.save(state, './checkpoint/{}/{}_current.pth'.format(args.dir, args.model))

	if accu > best_accu:
		best_accu = accu
		print('Saving best model: '+'./checkpoint/{}/{}_best.pth'.format(args.dir, args.model))
		torch.save(state, './checkpoint/{}/{}_best.pth'.format(args.dir, args.model))

# print('The best accuracy is: {:.4f}'.format(best_accu*100))
# if not args.resume:
# 	plt.ylabel('loss')
# 	plt.xlabel('epoch')
# 	plt.title("loss of {} changes with epochs".format(args.model))
# 	plt.plot(list(range(num_epochs)), record_loss, 'b', label='training loss')
# 	plt.plot(list(range(num_epochs)), record_test_loss, 'g', label='testing loss')
# 	plt.legend()
# 	plt.grid()
# 	plt.savefig('loss_{}_{}.png'.format(args.dir, args.model))
# plt.show()



