import argparse
import matplotlib.pyplot as plt

from common import *
from generator import Generator


def main():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--label', type=int, default=0, help='enter the label for the test.(Type:int, default : 0)')
    args = parser.parse_args()

    label = args.label

    model = Generator()
    model_file = osp.join(PRJ_DIR, 'model.pth')

    # loaded check pt;point
    loaded_check_pt = torch.load(model_file)

    # ckpt;check point , dict;dictionary
    ckpt_dict = {}

    # in case that training on deeplab repo.
    if 'state_dict' in loaded_check_pt.keys():
        loaded_check_pt = loaded_check_pt['state_dict']

    for item in loaded_check_pt.items():
        if item[0][:7] == "module.":
            new_key = item[0][7:]
        else:
            new_key = item[0]

        ckpt_dict[new_key] = item[1]

    model.load_state_dict(ckpt_dict)

    label_tensor = torch.zeros(10)
    label_tensor[label] = 1.0

    f, axarr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = model.forward(generate_random_seed(100), label_tensor)
            img = output.detach().numpy().reshape(28, 28)
            axarr[i, j].imshow(img, interpolation='none', cmap='Blues')
    plt.show()


if __name__ == '__main__':
    main()
