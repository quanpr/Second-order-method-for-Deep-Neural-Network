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


def line_search(z, d, tr):
	tau = (tr-comp_norm(z))/comp_norm(d)
	p = []
	for i in range(len(z)):
		p.append(z[i] + tau*d[i])

	while comp_norm(p) <= tr:
		tau *= 2
		# update on p
		for i in range(len(p)):
			p[i] = z[i] + tau*d[i]

	return tau/2

def trust_region_newton_cg(args, model, loss):
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
			prod += (grad_f[i]*d_j[i]).sum()

		# compute Hessian vector prod
		Hd = autograd.grad(prod, model.parameters(), create_graph=False, retain_graph=True)

		dTHd = 0.0
		for i in range(len(Hd)):
			dTHd += (Hd[i]*d_j[i]).sum()

		if dTHd <= 0.0:
			tau = line_search(z_j, d_j, args.tr)
			# update on model
			# pdb.set_trace()
			for (i, p) in enumerate(model.parameters()):
				p.data.add_(1, z_j[i] + tau*d_j[i])
			return loss, k+1

		r_j_norm = (comp_norm(r_j))**2
		alpha_j = r_j_norm/dTHd

		for i in range(len(z_j)):
			z_j[i] = z_j[i] + alpha_j * d_j[i]

		if comp_norm(z_j) >= args.tr:

			# Undo the update on z_j
			for i in range(len(z_j)):
				z_j[i] = z_j[i] - alpha_j * d_j[i]			
			# line search to find an appropriate tau
			tau = line_search(z_j, d_j, args.tr)
			# update
			# pdb.set_trace()
			for (i, p) in enumerate(model.parameters()):
				p.data.add_(1, z_j[i] + tau*d_j[i])
			return loss, k+1

		for i in range(len(r_j)):
			r_j[i] = r_j[i] + alpha_j*Hd[i]

		r_j_norm_update = comp_norm(r_j)
		if r_j_norm_update < args.eps:
			break

		beta_j = r_j_norm_update/r_j_norm

		for i in range(len(d_j)):
			d_j[i] = -r_j[i] + beta_j*d_j[i]

	# update on model
	for (i, p) in enumerate(model.parameters()):
		p.data.add_(1, z_j[i])
	return loss, k+1
