import matplotlib.pyplot as plt

def plot_single_example(image, mask, sampled_mask):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(mask)
    ax[1].set_title('Original Mask')
    ax[1].axis('off')
    ax[2].imshow(sampled_mask)
    ax[2].set_title('Sampled Mask')
    ax[2].axis('off')
    plt.show()