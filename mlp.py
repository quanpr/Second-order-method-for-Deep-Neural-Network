import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
	def __init__(self, hidden_dim=512, dropout=0.2):
		super(MLP, self).__init__()

		self.input_dim = 3*32*32
		self.hidden_dim = hidden_dim
		self.dropout = dropout

		self.FC_layers = nn.Sequential(
										nn.Linear(self.input_dim, hidden_dim),
										nn.BatchNorm1d(hidden_dim),
										nn.ReLU(inplace=True),

										nn.Linear(hidden_dim, hidden_dim),
										nn.BatchNorm1d(hidden_dim),
										nn.ReLU(inplace=True),

										nn.Linear(hidden_dim, hidden_dim),
										nn.BatchNorm1d(hidden_dim),
										nn.ReLU(inplace=True),
										# nn.Dropout(p=dropout),

										nn.Linear(hidden_dim, hidden_dim),
										nn.BatchNorm1d(hidden_dim),
										nn.ReLU(inplace=True),
										# nn.Dropout(p=dropout),

										nn.Linear(hidden_dim, hidden_dim),
										nn.BatchNorm1d(hidden_dim),
										nn.ReLU(inplace=True),
										# nn.Dropout(p=dropout),

										nn.Linear(hidden_dim, hidden_dim),
										nn.BatchNorm1d(hidden_dim),
										nn.ReLU(inplace=True),

										nn.Linear(hidden_dim, 10),
										)
		# self.softmax = nn.Softmax()


	def forward(self, x):
		N = x.shape[0]
		x = x.view(N, -1)

		out = self.FC_layers(x)

		return out


