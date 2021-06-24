import argparse
import src.core.archs as archs

from tqdm import tqdm
from src.utils.common import *
from src.utils.dataset import MnistDataset
from src.core.archs.discriminator import MnistDiscriminator
from src.core.archs.generator import MnistGenerator

def main():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--epoch', type=int, default=1, help='enter epoch for training. (type: int, default:1)')
    args = parser.parse_args()

    mnist_dataset = MnistDataset(osp.join(DATA_DIR, 'mnist_train.csv'))

    #d = archs.__dict__['MnistDiscriminator']
    #g = archs.__dict__['MnistGenerator']
    d = MnistDiscriminator()
    g = MnistGenerator()

    epochs = args.epoch

    for epoch in range(epochs):
        print("epoch = ", epoch + 1)
        for label, image_data_tensor, label_tensor in tqdm(mnist_dataset):
            # train discriminator on true
            d.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))

            random_label = generate_random_one_hot(10)
            # train discriminator on false
            # use detach() so gradients in G are not calculated
            d.train(g.forward(generate_random_seed(100), random_label).detach(), random_label, torch.FloatTensor([0.0]))

            random_label = generate_random_one_hot(10)

            # train generator
            g.train(d, generate_random_seed(100), random_label, torch.FloatTensor([1.0]))

    torch.save({'state_dict': g.state_dict()}, osp.join(PRJ_DIR, 'models/model_{}.pth'.format(epochs)))


if __name__ == '__main__':
    main()
