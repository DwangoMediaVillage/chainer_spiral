import numpy as np
import cv2
import glob
import os
import pandas

def fill(fg, imsize, factor=0.8):
    bg = np.ones((imsize, imsize), dtype=np.uint8)

    h, w = fg.shape
    scale = imsize / max(h, w) * factor
    fg = cv2.resize(fg, None, fx=scale, fy=scale)
    h, w = fg.shape

    c = int(imsize / 2)

    bg[c-int(h/2):c-int(h/2)+h, c-int(w/2):c-int(w/2)+w] = fg
    return bg

def convert_jikei_img(img, imsize, thres=190):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (H, W)
    _, img = cv2.threshold(img, thres, 1.0, cv2.THRESH_BINARY)

    # assuming the background value is 1
    img = fill(img, imsize)

    return img

def cut_by_density(tag, img, lower, higher):
    # assume that img is binary matrices [0, 1]
    # import ipdb; ipdb.set_trace()
    p = [ x.sum() / np.prod(x.shape) for x in img ]
    inds = np.argsort(np.array(p))
    N = len(inds)
    inds = inds[int(N * lower):int(N * (1.0-higher))]
    return [ img[ind] for ind in inds ], [ tag[ind] for ind in inds ]

def load_jikei(target_dir, tags, imsize, lower=0.05, higher=0.05):
    target_dir = os.path.join(target_dir, 'characters')

    r_tag = []
    r_img = []

    for tag in tags:
        tag = f"U+{tag}"

        # for each character
        target_dir_tag = os.path.join(target_dir, tag)
        if os.path.exists(target_dir_tag):
            # get all jpg filenames
            jpg_files = glob.glob(os.path.join(target_dir_tag, '*.jpg'))
        
            for jpg_file in jpg_files:
                # for each jpg file
                img = cv2.imread(jpg_file)  # (H, W, K), K = 3
                img = convert_jikei_img(img, imsize)

                r_tag.append(tag)
                r_img.append(img.astype(np.float32))
        else:
            print(f"tag {tag} does not exist in target_dir")
    
    r_img, r_tag = cut_by_density(r_tag, r_img, lower, higher)

    return r_img, r_tag

def load_tags(filename):
    tags = pandas.read_csv(filename, delimiter='\n', header=None)
    return list(tags[0])
