import argparse
import src.core.archs as archs

from tqdm import tqdm
from src.utils.common import *
from src.utils.dataset import MnistDataset
from src.core.archs.discriminator import MnistDiscriminator
from src.core.archs.generator import MnistGenerator


def main():
    model_list = [osp.splitext(ele)[0] for ele in sorted(os.listdir(MODEL_DIR)) if ele.endswith('.pth')]

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--epoch', type=int, default=1, help='enter epoch for training. (type: int, default:1)')
    parser.add_argument('--resume', type=str, default=None, choices=model_list,
                        help='(optional) choose from {}'.format(model_list))
    args = parser.parse_args()

    mnist_dataset = MnistDataset(osp.join(DATA_DIR, 'mnist_train.csv'))

    #d = archs.__dict__['MnistDiscriminator']
    #g = archs.__dict__['MnistGenerator']
    d = MnistDiscriminator()
    g = MnistGenerator()

    epochs = args.epoch
    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(osp.join(MODEL_DIR, args.resume))
        start_epoch = checkpoint['epoch'] + 1
        d.model.load_state_dict(checkpoint['d_model'])
        g.model.load_state_dict(checkpoint['g_model'])
        d.optimizer.load_state_dict(checkpoint['d_optimizer'])
        g.optimizer.load_state_dict(checkpoint['g_optimizer'])

        if epochs < start_epoch:
            print('[resume error] setting epoch is higher than saved')
        else:
            print('=> [resume] start from saved {}\n'.format(start_epoch))

    for epoch in range(start_epoch, epochs):
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

        torch.save({'epoch': epoch,
                    'd_model': d.model.state_dict(),
                    'g_model': g.model.state_dict(),
                    'd_optimizer': d.optimizer.state_dict(),
                    'g_optimizer': g.optimizer.state_dict()
                    }, osp.join(PRJ_DIR, 'models/latest_model.pth'))


if __name__ == '__main__':
    main()
