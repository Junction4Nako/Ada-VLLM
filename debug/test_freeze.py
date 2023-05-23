import torch
import torch.nn as nn

class test(nn.Module):
    def __init__(self, n=5):
        super(test, self).__init__()
        self.a = nn.Embedding(10, n)
        self.b = nn.Linear(n, n)
        self.c = nn.Linear(n, 10)

        self.a.weight = self.c.weight
        self.c.weight.requires_grad = False

    def forward(self, x):
        tmp = self.a(x)
        return self.c(self.b(tmp))

model = test()
model.train()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

k = torch.randint(0, 5, (3,))
ky = torch.randint(0, 5, (3, ))
ce = nn.CrossEntropyLoss()
optimizer.zero_grad()
pred = model(k)
loss = ce(pred, ky)
