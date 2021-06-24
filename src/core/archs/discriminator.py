import torch
import torch.nn as nn
import pandas


class MnistDiscriminator(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(784 + 10, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        # create loss function
        self.loss_function = nn.BCELoss()

        # create optimiser, simple stochastic gradient descent
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

    def forward(self, image_tensor, label_tensor):
        # 시드와 레이블 결합
        inputs = torch.cat((image_tensor, label_tensor))

        return self.model(inputs)

    def train(self, inputs, label_tensor, targets):
        # calculate the output of the network
        outputs = self.forward(inputs, label_tensor)

        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        # if self.counter % 10000 == 0:
        #     print("counter = ", self.counter)

        # zero gradients, perform a backward pass, update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=0, figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))

