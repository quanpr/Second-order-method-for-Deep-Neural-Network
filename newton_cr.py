import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.utils as utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pdb

def comp_norm(para):
	norm = para[0].new(1).zero_().type_as(para[0])
	for p in para:
		norm = norm + p.norm()**2
	return ((norm**0.5)).detach()

def comp_diff(para1, para2):
	norm_diff = 0.0
	norm_sum = 0.0
	for k1, k2 in zip(para1, para2):
		error = k1.data - k2.data
		norm_diff += torch.sum(error * error)
		norm_sum += torch.sum(k1.data * k1.data)
	return (norm_diff / norm_sum)

def newton_step_cubic_regularization(args, model, loss):

	grad_para = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	# state the initial step
	norm = comp_norm(grad_para)
	d = []
	for i in range(len(grad_para)):
		d.append(grad_para[i].clone())
		d[-1] = args.init * d[-1] / norm

	for k in range(args.N1):
		grad_vect_prod = 0.0
		for i in range(len(grad_para)):
			grad_vect_prod += (grad_para[i]*d[i]).sum()

		# Hess_vect_prod = H*d
		Hess_vect_prod = autograd.grad(grad_vect_prod, model.parameters(), create_graph=False, retain_graph=True)

		# Hess_vect_prod = autograd.grad(grad_para, model.parameters(), grad_outputs=d, retain_graph=True)

		norm = comp_norm(d)

		for i in range(len(d)):
			d[i] = d[i] - args.step * (Hess_vect_prod[i] + grad_para[i] + args.rho*3*norm*d[i])
			# d[i] = args.step * (d[i] - (Hess_vect_prod[i] + grad_para[i] + args.rho*3*norm*d[i]))
	
	# if boundary == False:
	# pdb.set_trace()
	if True:
		# update on model
		for idx, p in enumerate(model.parameters()):
			p.data.add_(1, d[idx])
		return loss, k+1

def newton_step_cr_cg(args, model, loss):

	grad_f = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	z_j, r_j, d_j = [], [], []
	# initialize z_j, r_j, d_j
	for i in range(len(grad_f)):
		z_j.append(torch.zeros(grad_f[i].shape).type_as(grad_f[i]))
		r_j.append(grad_f[i].clone())
		d_j.append(-grad_f[i].clone())

	for k in range(args.N1):
		prod = 0.0
		for i in range(len(grad_f)):
			prod = prod + (grad_f[i]*d_j[i]).sum()

		d_norm = comp_norm(d_j)
		# compute Hessian vector prod
		Hd = autograd.grad(prod, model.parameters(), create_graph=False, retain_graph=True)

		# The real: Hv = Hd + rho*d_norm*d
		dTHd = args.rho/3*d_norm**3
		# pdb.set_trace()
		for i in range(len(Hd)):
			# Hd[i] = Hd[i] + args.rho*d_norm*d_j[i]
			dTHd += (Hd[i]*d_j[i]).sum()

		if dTHd <= 0.0:
			# pdb.set_trace()
			if k == 0:
				# update on model
				for (i, p) in enumerate(model.parameters()):
					p.data.add_(1, args.step*d_j[i])
				return loss, k+1
			else:
				for (i, p) in enumerate(model.parameters()):
					p.data.add_(1, args.step*z_j[i])
				return loss, k+1

		r_j_norm = (comp_norm(r_j))**2
		alpha_j = r_j_norm/dTHd

		# update on z_j and r_j
		for i in range(len(z_j)):
			z_j[i] = z_j[i] + alpha_j * d_j[i]
			r_j[i] = r_j[i] + alpha_j * (Hd[i] + args.rho*d_norm*d_j[i])
			# r_j[i] = r_j[i] + alpha_j * Hd[i]

		# update on model if algorithm converges
		if comp_norm(r_j) <= args.eps:
			for (i, p) in enumerate(model.parameters()):
				p.data.add_(1, args.step*z_j[i])
			return loss, k+1			

		r_j_norm_update = comp_norm(r_j)
		beta_j = r_j_norm_update/r_j_norm

		for i in range(len(d_j)):
			d_j[i] = -r_j[i] + beta_j*d_j[i]

	# update on model
	for (i, p) in enumerate(model.parameters()):
		p.data.add_(1, args.step*z_j[i])
	return loss, k+1

gamma = 2
eta_up = 0.8
# eta_low = 0.1
eta_low = 1e-2

MAX = 1000
MIN = 1e-7
# MIN = 0

def newton_adaptive_cubic_regularization(args, model, inputs, targets, criterion, loss, data_full, target_full):

	args.rho = max(min(args.rho, MAX), MIN)
	# pdb.set_trace()
	loss_pre = criterion(model(data_full), target_full)
	grad_para_full = autograd.grad(loss_pre, model.parameters(), create_graph=False, retain_graph=False)
	loss_pre = loss_pre.item()
	torch.cuda.empty_cache()
	# pdb.set_trace()
	grad_para = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	# state the initial step
	norm = comp_norm(grad_para_full)
	d = []
	for i in range(len(grad_para_full)):
		d.append(grad_para_full[i].clone())
		d[-1] = args.init * d[-1] / norm

	for k in range(args.N1):
		grad_vect_prod = 0.0
		for i in range(len(grad_para)):
			grad_vect_prod += (grad_para[i]*d[i]).sum()

		# Hess_vect_prod = H*d
		Hess_vect_prod = autograd.grad(grad_vect_prod, model.parameters(), create_graph=False, retain_graph=True)
		# Hess_vect_prod = autograd.grad(grad_para, model.parameters(), grad_outputs=d, retain_graph=True)

		norm_d = comp_norm(d)

		for i in range(len(d)):
			d[i] = d[i] - args.step * (Hess_vect_prod[i] + grad_para_full[i] + args.rho*norm_d*d[i])
			# d[i] = args.step * (d[i] - (Hess_vect_prod[i] + grad_para[i] + args.rho*3*norm_d*d[i]))
	
	
	# update on model
	for idx, p in enumerate(model.parameters()):
		p.data.add_(1, d[idx])
	
	# evalute actual reduction v.s. predicted reduction
	# loss_update = criterion(model(inputs), targets)
	loss_update = (criterion(model(data_full), target_full)).item()
	torch.cuda.empty_cache()
	
	grad_vect_prod = 0.0
	for i in range(len(grad_para)):
		grad_vect_prod += (grad_para[i]*d[i]).sum()
	
	Hess_vect_prod = autograd.grad(grad_vect_prod, model.parameters(), create_graph=False, retain_graph=False)
	
	grad_vect_prod = 0.0
	for i in range(len(grad_para_full)):
		grad_vect_prod += ((grad_para_full[i]*d[i]).sum()).item()

	dTHd = 0.0
	for i in range(len(d)):
		dTHd += 1/2*((d[i] * Hess_vect_prod[i]).sum()).item()
	
	rho = (loss_pre - loss_update)/(-grad_vect_prod - dTHd - args.rho/3*(comp_norm(d))**3 )
	if rho >= eta_up:
		args.rho = max(args.rho/gamma, MIN)
		return loss_update, k+1
	# elif rho >= eta_low:
	# 	return loss_update, k+1
	else:
		# undo the update
		for i, p in enumerate(model.parameters()):
			p.data.add_(-1, d[i])
		args.rho = min(MAX, args.rho*gamma)
		return loss, k+1


def newton_adaptive_cr_cg(args, model, inputs, targets, criterion, loss, data_full, target_full):


	args.rho = max(min(args.rho, MAX), MIN)
	# pdb.set_trace()
	loss_pre = criterion(model(data_full), target_full)
	grad_f_full = autograd.grad(loss_pre, model.parameters(), create_graph=False, retain_graph=False)
	loss_pre = loss_pre.item()
	torch.cuda.empty_cache()

	grad_norm = comp_norm(grad_f_full).item()
	eps = min(0.5, grad_norm**0.5)*grad_norm

	grad_f = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	z_j, r_j, d_j = [], [], []
	# initialize z_j, r_j, d_j
	for i in range(len(grad_f_full)):
		z_j.append(torch.zeros(grad_f_full[i].shape).type_as(grad_f_full[i]))
		r_j.append(grad_f_full[i].clone())
		d_j.append(-grad_f_full[i].clone())

	nega_curve_k0 = False
	for k in range(args.N1):

		prod = 0.0
		for i in range(len(grad_f)):
			prod = prod + (grad_f[i]*d_j[i]).sum()

		d_norm = comp_norm(d_j)
		# compute Hessian vector prod
		Hd = autograd.grad(prod, model.parameters(), create_graph=False, retain_graph=True)

		# The real: Hv = Hd + rho*d_norm*d
		# dTHd = args.rho/3*d_norm**3
		dTHd = (args.rho*d_norm**2).item()
		
		for i in range(len(Hd)):
			# Hd[i] = Hd[i] + args.rho*d_norm*d_j[i]
			dTHd += ((Hd[i]*d_j[i]).sum()).item()

		if dTHd <= 0.0:
			if k == 0:
				# update on model
				# for (i, p) in enumerate(model.parameters()):
				# 	p.data.add_(1, args.step*d_j[i])
				# # return loss, k+1
				nega_curve_k0 = True
				break
			else:
				# for (i, p) in enumerate(model.parameters()):
				# 	p.data.add_(1, args.step*z_j[i])
				break

		r_j_norm = (comp_norm(r_j))**2
		alpha_j = r_j_norm/dTHd

		# pdb.set_trace()
		# update on z_j and r_j
		for i in range(len(z_j)):
			z_j[i] = z_j[i] + alpha_j * d_j[i]
			# r_j[i] = r_j[i] + alpha_j * (Hd[i] + args.rho*d_norm*d_j[i])
			r_j[i] = r_j[i] + alpha_j * (Hd[i] + args.rho*d_j[i])

		# pdb.set_trace()
		# update on model if algorithm converges
		if comp_norm(r_j) <= eps:
			# for (i, p) in enumerate(model.parameters()):
			# 	p.data.add_(1, args.step*z_j[i])
			break			


		r_j_norm_update = (comp_norm(r_j))**2
		beta_j = r_j_norm_update/r_j_norm

		for i in range(len(d_j)):
			d_j[i] = -r_j[i] + beta_j*d_j[i]


	# update on model
	if nega_curve_k0:
		for (i, p) in enumerate(model.parameters()):
			p.data.add_(1, args.step*d_j[i])		
	else:
		for (i, p) in enumerate(model.parameters()):
			p.data.add_(1, args.step*z_j[i])

	loss_update = (criterion(model(data_full), target_full)).item()
	torch.cuda.empty_cache()
	
	if nega_curve_k0:
		z_j = [d_j[i] for i in range(len(d_j))]
	del d_j

	prod = 0
	for i in range(len(grad_f)):
		prod += (grad_f[i]*z_j[i]).sum()
	Hd = autograd.grad(prod, model.parameters(), create_graph=False, retain_graph=False)
	
	prod = 0.0
	for i in range(len(grad_f_full)):
		prod += ((grad_f_full[i]*z_j[i]).sum()).item()
	dTHd_ = 0.0
	for i in range(len(z_j)):
		dTHd_ += ((z_j[i] * Hd[i]).sum()).item()
	
	pred_red = ((-prod - 1/2*dTHd_ - args.rho/2*(comp_norm(z_j))**2)).item()
	actual_red = (loss_pre - loss_update)
	rho = actual_red / pred_red
	# rho = (loss_pre - loss_update)/(-prod - dTHd_ - args.rho/3*(comp_norm(z_j))**3)

	# if actual_red < 0 and pred_red < 0:
	# 	pdb.set_trace()

	if rho >= eta_up and actual_red > 0:
		args.rho = max(args.rho/gamma, MIN)
		return loss_update, k+1, actual_red, pred_red, rho, eps
	elif rho >= eta_low and actual_red > 0:
		return loss_update, k+1, actual_red, pred_red, rho, eps
	else:
		# undo the update
		for i, p in enumerate(model.parameters()):
			p.data.add_(-1, args.step*z_j[i])
		args.rho = min(MAX, args.rho*gamma)

	return loss, k+1, actual_red, pred_red, rho, eps


# def newton_adaptive_cr_cg(args, model, inputs, targets, criterion, loss, data_full, target_full):


# 	args.rho = max(min(args.rho, MAX), MIN)
# 	# pdb.set_trace()
# 	loss_pre = criterion(model(data_full), target_full)
# 	grad_f_full = autograd.grad(loss_pre, model.parameters(), create_graph=False, retain_graph=False)
# 	loss_pre = loss_pre.item()
# 	torch.cuda.empty_cache()

# 	grad_f = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
# 	z_j, r_j, d_j = [], [], []
# 	# initialize z_j, r_j, d_j
# 	for i in range(len(grad_f_full)):
# 		z_j.append(torch.zeros(grad_f_full[i].shape).type_as(grad_f_full[i]))
# 		r_j.append(grad_f_full[i].clone())
# 		d_j.append(-grad_f_full[i].clone())

# 	nega_curve_k0 = False
# 	for k in range(args.N1):

# 		d_norm = comp_norm(d_j)

# 		Hd = []
# 		for i in range(len(d_j)):
# 			Hd.append(10/(i+1)*d_j[i].clone())

# 		dTHd = 0.0
# 		# pdb.set_trace()
# 		for i in range(len(Hd)):
# 			# Hd[i] = Hd[i] + args.rho*d_norm*d_j[i]
# 			dTHd += (Hd[i]*d_j[i]).sum()

# 		if dTHd <= 0.0:
# 			# pdb.set_trace()
# 			if k == 0:
# 				# update on model
# 				# for (i, p) in enumerate(model.parameters()):
# 				# 	p.data.add_(1, args.step*d_j[i])
# 				# # return loss, k+1
# 				nega_curve_k0 = True
# 				break
# 			else:
# 				# for (i, p) in enumerate(model.parameters()):
# 				# 	p.data.add_(1, args.step*z_j[i])
# 				break

# 		r_j_norm = (comp_norm(r_j))**2
# 		alpha_j = r_j_norm/dTHd

		
# 		# update on z_j and r_j
# 		for i in range(len(z_j)):
# 			z_j[i] = z_j[i] + alpha_j * d_j[i]
# 			# r_j[i] = r_j[i] + alpha_j * (Hd[i] + args.rho*d_norm*d_j[i])
# 			r_j[i] = r_j[i] + alpha_j * (Hd[i])

# 		pdb.set_trace()
# 		# update on model if algorithm converges
# 		if comp_norm(r_j) <= args.eps:
# 			# for (i, p) in enumerate(model.parameters()):
# 			# 	p.data.add_(1, args.step*z_j[i])
# 			break			

# 		r_j_norm_update = (comp_norm(r_j))**2
# 		beta_j = r_j_norm_update/r_j_norm

# 		for i in range(len(d_j)):
# 			d_j[i] = -r_j[i] + beta_j*d_j[i]

# 		pdb.set_trace()


# 	# update on model
# 	if nega_curve_k0:
# 		for (i, p) in enumerate(model.parameters()):
# 			p.data.add_(1, args.step*d_j[i])		
# 	else:
# 		for (i, p) in enumerate(model.parameters()):
# 			p.data.add_(1, args.step*z_j[i])

# 	loss_update = (criterion(model(data_full), target_full)).item()
# 	torch.cuda.empty_cache()
	
# 	if nega_curve_k0:
# 		z_j = [d_j[i] for i in range(len(d_j))]
# 	del d_j

# 	prod = 0
# 	for i in range(len(grad_f)):
# 		prod += (grad_f[i]*z_j[i]).sum()
# 	Hd = autograd.grad(prod, model.parameters(), create_graph=False, retain_graph=False)
	
# 	prod = 0.0
# 	for i in range(len(grad_f_full)):
# 		prod += ((grad_f_full[i]*z_j[i]).sum()).item()
# 	dTHd = 0.0
# 	for i in range(len(z_j)):
# 		dTHd += ((z_j[i] * Hd[i]).sum()).item()
	
# 	rho = (loss_pre - loss_update)/(-prod - 1/2*dTHd - args.rho/2*(comp_norm(z_j))**2)
# 	# rho = (loss_pre - loss_update)/(-prod - dTHd - args.rho/3*(comp_norm(z_j))**3)
# 	# pdb.set_trace()
# 	if rho >= eta_up:
# 		args.rho = max(args.rho/gamma, MIN)
# 		return loss_update, k+1
# 	elif rho >= eta_low:
# 		return loss_update, k+1
# 	else:
# 		# undo the update
# 		for i, p in enumerate(model.parameters()):
# 			p.data.add_(-1, args.step*z_j[i])
# 		args.rho = min(MAX, args.rho*gamma)

# 	return loss, k+1
