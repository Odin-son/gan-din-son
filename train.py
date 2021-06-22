import os.path as osp

from common import *
from dataset import MnistDataset
from discriminator import Discriminator
from generator import Generator


def main():
    mnist_dataset = MnistDataset(osp.join(DATA_DIR, 'mnist_train.csv'))

    d = Discriminator()
    g = Generator()

    epochs = 12
    for epoch in range(epochs):
        print("epoch = ", epoch + 1)

        for label, image_data_tensor, label_tensor in mnist_dataset:
            # train discriminator on true
            d.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))

            random_label = generate_random_one_hot(10)
            # train discriminator on false
            # use detach() so gradients in G are not calculated
            d.train(g.forward(generate_random_seed(100), random_label).detach(), random_label, torch.FloatTensor([0.0]))

            random_label = generate_random_one_hot(10)

            # train generator
            g.train(d, generate_random_seed(100), random_label, torch.FloatTensor([1.0]))


if __name__ == '__main__':
    main()
