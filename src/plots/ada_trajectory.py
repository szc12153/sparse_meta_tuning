import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from collections import Counter

def plot_episode(x_s, y_s, x_q,
                 y_q, size_multiplier=1, max_imgs_per_col=10,
                 max_imgs_per_row=10, filename="test_episode"):
    x_s, y_s, x_q, y_q = x_s.cpu().numpy(), y_s.cpu().numpy(), x_q.cpu().numpy(), y_q.cpu().numpy()
    for name, images, class_ids in zip(('Support', 'Query'),
                                     (x_s, x_q),
                                     (y_s, y_q)):
        n_samples_per_class = Counter(class_ids)
        n_samples_per_class = {k: min(v, max_imgs_per_col)
                               for k, v in n_samples_per_class.items()}
        id_plot_index_map = {k: i for i, k
                             in enumerate(n_samples_per_class.keys())}
        num_classes = min(max_imgs_per_row, len(n_samples_per_class.keys()))
        max_n_sample = max(n_samples_per_class.values())
        figwidth = max_n_sample
        figheight = num_classes
        if name == 'Support':
            print('#Classes: %d' % len(n_samples_per_class.keys()))
        figsize = (figheight * size_multiplier, figwidth * size_multiplier)
        fig, axarr = plt.subplots(figwidth, figheight, figsize=figsize)
        fig.suptitle('%s Set' % name, size='20')
        fig.tight_layout(pad=3, w_pad=0.1, h_pad=0.1)
        reverse_id_map = {v: k for k, v in id_plot_index_map.items()}
        for i, ax in enumerate(axarr.flat):
            ax.patch.set_alpha(0)
            # Print the class ids, this is needed since, we want to set the x axis
            # even there is no picture.
            ax.set(xlabel=reverse_id_map[i % figheight], xticks=[], yticks=[])
            ax.label_outer()
        for image, class_id in zip(images, class_ids):
            # First decrement by one to find last spot for the class id.
            n_samples_per_class[class_id] -= 1
            # If class column is filled or not represented: pass.
            if (n_samples_per_class[class_id] < 0 or id_plot_index_map[class_id] >= max_imgs_per_row):
                continue
            # If width or height is 1, then axarr is a vector.
            if axarr.ndim == 1:
                ax = axarr[n_samples_per_class[class_id] if figheight == 1 else id_plot_index_map[class_id]]
            else:
                ax = axarr[n_samples_per_class[class_id], id_plot_index_map[class_id]]
            ax.imshow(np.transpose(image,(1,2,0)))

        fig.savefig(f"figures/{filename}_{name}.png",dpi=250)
        plt.close(fig)


def save_raw_pkl(dict_file:dict, path:str, savename:str):
    if not savename.endswith(".pkl"):
        savename += ".pkl"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, savename)
    with open(filename, 'wb') as fp:
        pickle.dump(dict_file, fp)


def plt_and_save_adaptation_trajectory(trajecotory, path:str, savename:str):
    if not os.path.exists(path):
        os.makedirs(path)

    fig = plt.figure()
    x_steps = list(range(len(trajecotory[0])))
    plt.plot(x_steps,trajecotory[0],":")
    plt.fill_between(x_steps, trajecotory[0]-trajecotory[1],trajecotory[0]+trajecotory[1], alpha=0.2)
    filename = os.path.join(path, savename)
    fig.savefig(filename)
    plt.close(fig)
    with open(filename.replace('.png','.pkl'), 'wb') as fp:
        pickle.dump({"mean":trajecotory[0],
                     "conf": trajecotory[1]}, fp)

