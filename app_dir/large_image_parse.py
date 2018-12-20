from build_model import TextClassifier
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import os
import numpy as np


def break_images(image):
    k, y, x, rgb_in = image.shape
    im = image[0]

    x_break = (x - x%256)/256
    y_break = (y - y%256)/256
    im_list = []
    im_idx = []
    for j in range(256, y - y%256 + 256, 256):
        for i in range(256, x - x%256 + 256, 256):
            im_list.append(im[j-256:j,i-256:i])
            im_idx.append(np.array([[i-256,j-256],[i,j]]))

    return im_list, im_idx

def predict_on_list(im_list, model):

    model._make_predict_function()

    pred = []
    count = 0
    for i in im_list:
        if i.shape[0]==256 & i.shape[1]==256:
            # print('predicting..',count)
            pred.append(model.predict(np.expand_dims(i, axis =0)))

        else:
            # print('skipped...',count)
            # print('Shape not compatable at the moment: ',i.shape)
            pred.append(-1)
        count = count + 1

    return pred


def plot_predictions(img, predictions, im_idx, file_save='static/images_predicted/test_image' ):
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    for idx, i in enumerate(predictions):
        # print('making rectangle: ',idx, ' | proba: ',i)
        # Create a Rectangle patch
        if i != -1:
            x = im_idx[idx][0][0]
            y = im_idx[idx][0][1]
            if i >= 0.75:
                rect = patches.Rectangle((x,y),256,256,linewidth=1,edgecolor='r',facecolor='none')
                # ax.text(x + 15, y+ 15, 'Crack', fontsize=12)
            else:
                rect = patches.Rectangle((x,y),256,256,linewidth=1,edgecolor='b',facecolor='none')
                # ax.text(x + 15, y+ 15, 'NO Crack', fontsize=12)
            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.savefig(file_save)
    # plt.show()


def run_all(model, img, file_save='static/images_predicted/test_image'):
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    im_list, im_idx = break_images(img_tensor)

    predictions = predict_on_list(im_list, model)
    plot_predictions(img, predictions, im_idx,file_save=file_save)
    pass



if __name__ == '__main__':

    model = load_model('static/final_model_12_15_18_1.hdf5')


    #img = image.load_img('static/images_f/20181219_154206.jpg')
    #img = image.load_img('static/images_f/20181219_154203.jpg')
    img = image.load_img('static/images_f/20181219_154209.jpg')
    #img = image.load_img('/home/smw/Documents/galvanize/capstone_zone/crack_detection/data/test_train_hold_1/hold/NO_crack/001-23.jpg')
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    im_list, im_idx = break_images(img_tensor)

    predictions = predict_on_list(im_list, model)
    plot_predictions(predictions, im_idx)
