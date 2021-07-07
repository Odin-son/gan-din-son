import argparse
import matplotlib.pyplot as plt

from src.utils.common import *
from src.core.archs.generator import MnistGenerator, CelebAGenerator


def main():

    model_list = [osp.splitext(ele)[0] for ele in sorted(os.listdir(MODEL_DIR)) if ele.endswith('.pth')]

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--model', type=str, choices=model_list, help='choose from {}'.format(model_list),
                        required=True)
    parser.add_argument('--label', type=int, default=0, help='enter the label for the test.(Type:int, default : 0)')
    args = parser.parse_args()

    label = args.label

    model = CelebAGenerator()
    model_file = osp.join(MODEL_DIR, '{}.pth'.format(args.model))

    # loaded check pt;point
    loaded_check_pt = torch.load(model_file)

    # ckpt;check point , dict;dictionary
    ckpt_dict = {}

    # in case that training on deeplab repo.
    if 'g_model' in loaded_check_pt.keys():
        loaded_check_pt = loaded_check_pt['g_model']

    for item in loaded_check_pt.items():
        if item[0][:7] == "module.":
            new_key = item[0][7:]
        else:
            new_key = "model."+item[0]

        ckpt_dict[new_key] = item[1]

    model.load_state_dict(ckpt_dict)

    f, axarr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = model.forward(generate_random_seed(100))
            img = output.detach().permute(0,2,3,1).view(128,128,3).numpy()
            axarr[i, j].imshow(img, interpolation='none', cmap='Blues')
    plt.show()


if __name__ == '__main__':
    main()
