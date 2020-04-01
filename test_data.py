from data import *
import matplotlib.pyplot as plt


def test_single_image():
    view_matrix, projection_matrix = init(0)

    seg_img, _, _, _ = generate_random_object(view_matrix, projection_matrix)
    hw = seg_img.shape[0]
    seg_img = seg_img.reshape(hw, hw)
    print("{0:.2f}%".format(100 * np.sum(seg_img == 1) / (hw * hw)))
    plt.imshow(seg_img)
    plt.show()


def test_image_in_dataset(i):
    input, label = load_data(data_filename)
    print(input.shape)
    img = input[i].reshape(input.shape[2], input.shape[3])
    print("{0:.2f}%".format(100 * np.sum(img == 1) / (input.shape[2] * input.shape[3])))
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # test_single_image()
    data_filename = 'data_test.h5'
    for i in range(16):
        test_image_in_dataset(i)
