import argparse

from tqdm import tqdm
from src.utils.common import *
from src.utils.dataset import MnistDataset, CelebADataset
from src.core.archs.discriminator import MnistDiscriminator, CelebADiscriminator
from src.core.archs.generator import MnistGenerator, CelebAGenerator


def main():
    model_list = [osp.splitext(ele)[0] for ele in sorted(os.listdir(MODEL_DIR)) if ele.endswith('.pth')]

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--epoch', type=int, default=1, help='enter epoch for training. (type: int, default:1)')
    parser.add_argument('--resume', type=str, default=None, choices=model_list,
                        help='(optional) choose from {}'.format(model_list))
    args = parser.parse_args()

    dataset = CelebADataset(osp.join(DATA_DIR, 'celeba_aligned_small.h5py'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    d = CelebADiscriminator()
    g = CelebAGenerator()
    d.to(device)
    g.to(device)

    epochs = args.epoch
    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(osp.join(MODEL_DIR, '{}.pth'.format(args.resume)))
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
        for image_data_tensor in tqdm(dataset):
            # train discriminator on true
            d.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))

            # train discriminator on false
            # use detach() so gradients in G are not calculated
            d.train(g.forward(generate_random_seed(100)).detach(), torch.cuda.FloatTensor([0.0]))

            # train generator
            g.train(d, generate_random_seed(100), torch.cuda.FloatTensor([1.0]))

        torch.save({'epoch': epoch,
                    'd_model': d.model.state_dict(),
                    'g_model': g.model.state_dict(),
                    'd_optimizer': d.optimizer.state_dict(),
                    'g_optimizer': g.optimizer.state_dict()
                    }, osp.join(PRJ_DIR, 'models/latest_model.pth'))
    torch.save({'epoch': epoch,
                'd_model': d.model.state_dict(),
                'g_model': g.model.state_dict(),
                'd_optimizer': d.optimizer.state_dict(),
                'g_optimizer': g.optimizer.state_dict()
                }, osp.join(PRJ_DIR, 'models/celeb_{}.pth'.format(epoch)))


if __name__ == '__main__':
    main()
