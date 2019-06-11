import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.utils as utils
import pdb

def comp_norm(para):
	norm = para[0].new(1).zero_().type_as(para[0])
	for p in para:
		norm = norm + p.norm()**2
	return (norm**0.5)

def Hv_exact(net, loss_f, input, output, v):
	loss  = loss_f(net(input), output)
	grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
	Hv = torch.autograd.grad(grad_params, net.parameters(), grad_outputs=v, only_inputs=True)
	return Hv

def trust_region_newton_step(args, model, loss):

	grad_para = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	# state the initial step
	boundary = False
	norm = comp_norm(grad_para).detach()
	d = []
	for i in range(len(grad_para)):
		d.append(grad_para[i].clone())
		d[-1] = args.init * d[-1] / norm

	for k in range(args.N1):
		if boundary == False:
			grad_vect_prod = 0.0
			for i in range(len(grad_para)):
				grad_vect_prod += (grad_para[i]*d[i]).sum()

			# Hess_vect_prod = H*d
			Hess_vect_prod = autograd.grad(grad_vect_prod, model.parameters(), create_graph=False, retain_graph=True)

			# Hess_vect_prod = autograd.grad(grad_para, model.parameters(), grad_outputs=d, retain_graph=True)

			for i in range(len(d)):
				d[i] = d[i] - args.step * (Hess_vect_prod[i] + grad_para[i])

			norm = comp_norm(d)
			if norm >= args.tr:
				boundary = True
		else:
			break

	
	# if boundary == False:
	# pdb.set_trace()
	if True:
		# update on model
		idx = 0
		for p in model.parameters():
			# p.data.add_(1, d[idx])
			if boundary:
				p.data.add_(1, d[idx]/norm*args.tr)
			else:
				 p.data.add_(1, d[idx])
			idx += 1
		return loss, k+1

	
	for _ in range(args.N2):
		alpha = 1
		# cache d for backtracking
		# state = copy.deepcopy(model)

		# compute projection onto the boundary
		# Define: v = H*d + g
		# 		  product = d^T*v
		product = loss.new(1).zero_().type_as(loss)
		for i in range(len(Hess_vect_prod)):
			product += ((Hess_vect_prod[i]+grad_para[i])*d[i]).sum()

		# Define prod = (I -d*d^T)(H*d + g) = v - product * d
		# where product = d^T*v
		prod = []
		# Compute f(d) prior to update for backtracking line search
		f_d = loss.new(1).zero_().type_as(loss)
		for i in range(len(Hess_vect_prod)):
			prod.append(Hess_vect_prod[i] + grad_para[i] - d[i]*product)
			# compute f(d)
			f_d = f_d + ((0.5*Hess_vect_prod[i] + grad_para[i])*d[i]).sum()


		# find acceptable alpha by amijo line search
		# while True:
		for i in range(5):
			d_update = []
			descent = loss.new(1).zero_().type_as(loss)

			# compute d_update
			for i in range(len(d)):
				d_update.append(d[i] - alpha * prod[i])
			norm_d_update = comp_norm(d_update)
			for i in range(len(d_update)):
				d_update[i] = d_update[i] / norm_d_update

			# compute f_d_update through Hessian product estimation
			grad_vect_prod = loss.new(1).zero_().type_as(loss) 
			f_d_update = loss.new(1).zero_().type_as(loss) 
			for i in range(len(grad_para)):
				grad_vect_prod += (grad_para[i]*d_update[i]).sum()
			Hess_vect_prod = autograd.grad(grad_vect_prod, model.parameters(), create_graph=False, retain_graph=True)

			for i in range(len(d_update)):
				# compute f_d_update
				f_d_update = f_d_update + ((0.5*Hess_vect_prod[i] + grad_para[i])*d_update[i]).sum()

				# compute descent rate
				descent = descent + 0.5 * alpha * ((Hess_vect_prod[i] + grad_para[i])*(d_update[i]-d[i])).sum()


			if f_d_update <= f_d + descent:
				break
			else:
				alpha /= 2

		# update on model
		idx = 0
		for p in model.parameters():
			p.data.add_(1, d_update[idx])
			idx += 1

def gradient_checker(Hess_vect_prod, Hess_vect_prod_):
	norm_diff = 0.0
	norm_sum = 0.0
	for k1, k2 in zip(Hess_vect_prod, Hess_vect_prod_):
		error = k1.data - k2.data
		norm_diff += torch.sum(error * error)
		norm_sum += torch.sum(k1.data * k1.data)
	diff = norm_diff / norm_sum
	return diff

def Hv_approx(net, loss_f, input, output, v, scale=1e6):
	loss  = loss_f(net(input), output)
	grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=False)
	for p, vi in zip(net.parameters(), v):
		p.data += vi.data / scale
	loss  = loss_f(net(input), output)
	grad_params_ = torch.autograd.grad(loss, net.parameters(), create_graph=False)
	Hv = []
	for g1, g2 in zip(grad_params, grad_params_):
		Hv.append((g2 - g1) * scale)
	for p, vi in zip(net.parameters(), v):
		p.data -= vi.data / scale
	return Hv

def Hv_exact2(net, loss_f, input, output, v):
	loss  = loss_f(net(input), output)
	grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True)
	inner = 0
	for k1, k2 in zip(grad_params, v):
		inner += torch.sum(k1 * k2)
	Hv = torch.autograd.grad(inner, net.parameters())
	return Hv

def Hv_approx2(grad_params, net, loss_f, input, output, v, scale=1e6):
	#loss  = loss_f(net(input), output)
	#grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=False)
	for p, vi in zip(net.parameters(), v):
		p.data += vi / scale
	loss  = loss_f(net(input), output)
	grad_params_ = torch.autograd.grad(loss, net.parameters(), create_graph=False)
	Hv = []
	for g1, g2 in zip(grad_params, grad_params_):
		Hv.append((g2 - g1) * scale)
	for p, vi in zip(net.parameters(), v):
		p.data -= vi / scale
	return Hv

def trust_region_newton_step2(args, model, loss, criterion, inputs, outputs, hidden=torch.tensor([0])):
	if args.model == 'RNN' and hidden == torch.tensor([0]):
		raise ValueError('need to provide hidden state')
	grad_para = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	# state the initial step
	boundary = False
	norm = comp_norm(grad_para).detach()
	d = []
	for i in range(len(grad_para)):
		d.append(grad_para[i].clone())
		d[-1] = args.init * d[-1] / norm
	
	for k in range(args.N1):
		if boundary == False:
			# grad_vect_prod = 0.0
			# for i in range(len(grad_para)):
			# 	grad_vect_prod += (grad_para[i]*d[i]).sum()

			# # Hess_vect_prod = H*d
			# Hess_vect_prod = autograd.grad(grad_vect_prod, model.parameters(), create_graph=False, retain_graph=True)

			Hess_vect_prod = Hv_approx2(grad_para, model, criterion, inputs, outputs, d)
			# Hess_vect_prod_ = autograd.grad(grad_para, model.parameters(), grad_outputs=d, retain_graph=True)

			# # diff = loss.new(1).zero_().type_as(loss)
			# diff = gradient_checker(Hess_vect_prod, Hess_vect_prod_)
			# pdb.set_trace()

			for i in range(len(d)):
				d[i] = d[i] - args.step * (Hess_vect_prod[i] + grad_para[i])

			norm = comp_norm(d)
			if norm >= args.tr:
				boundary = True
		else:
			break

	
	# if boundary == False:
	if True:
		# clip the update
		utils.clip_grad_norm_(d, args.tr)
		# update on model
		idx = 0
		for p in model.parameters():
			p.data.add_(1, d[idx]) 
			idx += 1
		return 

	
	for _ in range(args.N2):
		# compute projection onto the boundary
		Hess_vect_prod = Hv_approx2(grad_para, model, criterion, inputs, outputs, d)

		for i in range(len(d)):
			d[i] = d[i] - args.step * (Hess_vect_prod[i] + grad_para[i])

		norm = comp_norm(d)
		if norm >= args.tr:
			for i in range(len(d)):
				d[i] = d[i]/(norm+1e-9)

	# update on model
	idx = 0
	for p in model.parameters():
		p.data.add_(1, d[idx])
		idx += 1

gamma = 4/3
eta = 3/4

def adaptive_trust_region_newton_step(args, model, inputs, targets, trust_region, loss, criterion, hidden=torch.tensor([0])):
	
	trust_region = trust_region

	if args.model == 'RNN' and hidden == torch.tensor([0]):
		raise ValueError('need to provide hidden state')
	grad_para = autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	# state the initial step
	boundary = False
	norm = comp_norm(grad_para).detach()

	for itr in range(args.N2):
		d = []
		for i in range(len(grad_para)):
			d.append(grad_para[i].clone())
			d[-1] = args.init * d[-1] / norm
		for _ in range(args.N1):
			if boundary == False:
				grad_vect_prod = loss.new(1).zero_().type_as(loss)
				for i in range(len(grad_para)):
					grad_vect_prod += (grad_para[i]*d[i]).sum()

				# Hess_vect_prod = H*d
				Hess_vect_prod = autograd.grad(grad_vect_prod, model.parameters(), create_graph=False, retain_graph=True)

				for i in range(len(d)):
					d[i] = d[i] - args.step * (Hess_vect_prod[i] + grad_para[i])

				norm = comp_norm(d)
				if norm >= trust_region:
					boundary = True
			else:
				break

		# compute f_d_update through Hessian product estimation
		grad_vect_prod = loss.new(1).zero_().type_as(loss) 
		f_d_update = loss.new(1).zero_().type_as(loss) 
		for i in range(len(grad_para)):
			grad_vect_prod += (grad_para[i]*d[i]).sum()
		Hess_vect_prod = autograd.grad(grad_vect_prod, model.parameters(), create_graph=False, retain_graph=True)

		for i in range(len(d)):
			# compute f_d_update
			f_d_update = f_d_update + ((0.5*Hess_vect_prod[i] + grad_para[i])*d[i]).sum()

		# compute loss_update and get updated model
		idx = 0
		for p in model.parameters():
			p.data.add_(1, d[idx])
			idx += 1

		loss_update = criterion(model(inputs), targets)

		if (loss - loss_update)/(-f_d_update) > eta:
			trust_region *= gamma
			break
		elif (loss - loss_update)/(-f_d_update) > 1-eta:
			break
		else:
			idx = 0
			for p in model.parameters():
				p.data.add_(-1, d[idx])
				idx += 1			
			trust_region /= gamma

	return trust_region







	
