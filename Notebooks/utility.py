import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity

# Function to crop image 
def crop(img, mask=False):
    #gaussian filter
    tmp = cv.GaussianBlur(img,(21,21),0)

    ret1, th1 = cv.threshold(tmp, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret2, th2 = cv.threshold(img, 120, 255, cv.THRESH_BINARY)

    x, y, w, h = cv.boundingRect(th1)

    if(w<h):
        diff = h-w
        y = y+(diff//2)
        h = h-diff
    #return input image with crop and in b&w
    border = 50
    if(mask):
        return th2[y-border:y+h+border, x-border:x+w+border]
    return img[y-border:y+h+border, x-border:x+w+border]



# Function to scale pixel values from 0..255 to 0..1
def scale_pixels(img, scale_range=(0, 1)):
    return img.astype('float32') / 255.


# Function to resize images
def resize(img, size=(1024, 1024)):
    return cv.resize(img, size)


# Function to rotate an image
def rotate_image(img, angle):
  image_center = tuple(np.array(img.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
  return result


# Function to change brightness and contrast of an image
def brightness_contrast(img, brightness, contrast):
    img = np.int16(img)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


# Function to divide images into smaller parts
def split_patches(img, n_patches=16):
    imgs_list = []
    dim = int(img.shape[0]//(16**(1/2)))
    for i in range(0, n_patches):
        x = int(dim*(i%(16**(1/2))))
        y = int(dim*(i//(16**(1/2))))
        imgs_list.append(img[y:y+dim, x:x+dim])
    return imgs_list


# Function to summarize history for loss
def plot_hist(hist, title):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("LOSS " + title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
   
    
# Funtion to load images (all without defects)
def data_loader(dir='./', scale=False):
    imgs = []
    for img in os.listdir(dir):
        image = cv.imread((dir + '/' + img), cv.IMREAD_GRAYSCALE)
        if scale:
            image = image.astype('float32') / 255.
        imgs.append(image)
    return imgs

def calc_image_diff(true_img, rec_img):    
    # All values between 0 and 1
    rec_img[np.where((rec_img > [1]))] = 1
    rec_img[np.where((rec_img < [0]))] = 0
    # Calculate the difference between the original image and the reconstructed one
    diff = cv.absdiff(true_img, rec_img)
    # Scale values from 0..1 to 0..255 (required for opencv functions)
    diff = (diff*255).astype('uint8')
    # Calculate the SSIM similiarity score between the 2 slice
    score, _ = structural_similarity(true_img, rec_img, full=True)
    return diff, score


def predict_image(img, model):
    # Preprocess
    img = crop(img)
    img = resize(img)
    img = scale_pixels(img)
    slices = split_patches(img)
    slices_anomal_bound = []
    # Get the reconstructed images from the model
    ssim_score = []
    for slice in slices:
        slice_rec = model.predict(np.expand_dims(np.expand_dims(slice, 0), -1))
        slice_rec = np.squeeze(slice_rec[0], -1)
        diff, score = calc_image_diff(slice, slice_rec)
        slices_anomal_bound.append(diff)
        ssim_score.append(score)
    # Mean SSIM score, to understand if the autoencoder was able to reconstruct correctly the image
    ssim_score = np.mean(ssim_score)
    # Reconstruct the full image difference 
    horizion_concat = []
    for i in range(0, 16, 4):
        horizion_concat.append(cv.hconcat([slices_anomal_bound[i], slices_anomal_bound[i+1], slices_anomal_bound[i+2], slices_anomal_bound[i+3]]))
    img_total_anom = cv.vconcat([horizion_concat[0], horizion_concat[1], horizion_concat[2], horizion_concat[3]])
    # Detect and localize possible anomalies
    anomaly, anomalies_img = localize_anom(img, img_total_anom)
    return anomaly, anomalies_img, ssim_score

def localize_anom(img, diff):
    anomaly = False
    # Threshold to make the image only white and black (so higlight the anomalies in white)
    ret, th = cv.threshold(diff, 150, 255, cv.THRESH_BINARY)
    # Find the countours of the anomalies
    cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # For every anomaly, draw the bounding box
    anomalies_img = img.copy()
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        anomalies_img = cv.rectangle(img, (x-20, y-20), (x + w + 20, y + h + 20), (0, 255, 0), 2)
        # If some anomaly is found, we classify the image as anomalous
        anomaly=True
    return anomaly, anomalies_img

def calc_image_diff(true_img, rec_img):    
    # All values between 0 and 1
    rec_img[np.where((rec_img > [1]))] = 1
    rec_img[np.where((rec_img < [0]))] = 0
    # Calculate the difference between the original image and the reconstructed one
    diff = cv.absdiff(true_img, rec_img)
    # Scale values from 0..1 to 0..255 (required for opencv functions)
    diff = (diff*255).astype('uint8')
    # Calculate the SSIM similiarity score between the 2 slice
    score, _ = structural_similarity(true_img, rec_img, full=True)  
    return diff, score