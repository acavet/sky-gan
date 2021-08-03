'''Scrapes n images from given subreddit using Reddit API, formats for GAN training.
Not limited by usual 1k image cap.'''

# import praw
import os
import urllib
from PIL import Image
from io import BytesIO
import requests
import time
import math

imgs_path = '/Volumes/SKYFLASH/fromReddit/sky/'  # Where to save imgs
subreddit = 'SkyPorn'
n_imgs = 10000  # Number images to retrieve
allowed_extensions = ['.jpg', '.jpeg']
img_size = 32  # Pxl dimension of processed square image


def add_pushshift_100(subreddit, allowed_extensions, n_to_get, before):
    '''Get 100 image links from before certain timestamp with correct extensions
    Return: image links, earliest retrieved post timestamp (for next round)'''
    print(
        f'Getting up to {n_to_get} more image links with the right extensions...')
    url = f'https://api.pushshift.io/reddit/search/submission/?size={ n_to_get }&before={ str(before) }+&subreddit={ str(subreddit) }'
    print(url)
    data = requests.get(url).json()['data']
    print(data)
    # Get img links with right extensions
    img_list = []
    for post in data:
        # Depending on post could have different image url names
        try:
            if os.path.splitext(post['url_overridden_by_dest'])[-1] in allowed_extensions:
                img_list.append(post['url_overridden_by_dest'])
        except:
            pass
        try:
            if os.path.splitext(post['url'])[-1] in allowed_extensions:
                img_list.append(post['url'])
        except:
            pass
    # img_links = [post['url_overridden_by_dest'] for post in data
    #              if os.path.splitext(post['url_overridden_by_dest'])[-1] in allowed_extensions]
    return {'img_list': img_list, 'curr_unix': data[-1]['created_utc']}


def get_n_pics(n, subreddit, allowed_extensions=['.jpg', '.jpeg'], before=math.floor(time.time())):
    '''Retrieve n image links from a subreddit (most recent, with specified extensions'''
    n_left = n
    curr_unix = before
    img_list = []
    while n_left > 0:
        n_to_get = 100 if n_left > 100 else n_left
        results = add_pushshift_100(
            subreddit, allowed_extensions, n_to_get, curr_unix)
        curr_unix = results['curr_unix']
        img_list = img_list + results['img_list']
        n_left = n_left-len(results['img_list'])
        print(
            f'\n\n Need {n_left} more images, currently have {len(img_list)} image links: {img_list}')
    print(f'\n\nRetrieved {len(img_list)} images!')
    return img_list


# List of all sky img links
final_img_links = get_n_pics(n_imgs, subreddit, ['.jpg', '.jpeg'])


def process_img(img, img_path):
    '''Crops downloaded img to square, reduces size'''
    try:
        im = Image.open(img)
        w, h = im.size
        sq_len = min(w, h)
        sq_im = im.crop((0, 0, sq_len, sq_len))
        sq_small_im = sq_im.resize((img_size, img_size))
        sq_small_im.save(img_path)
    except:
        print('Something went wrong while processing the above image, skipped.')


n_errs = 0

# Download and process jpg/jpeg imgs
for index, url in enumerate(final_img_links):
    _, ext = os.path.splitext(url)
    try:
        download_path = imgs_path + \
            str(index) + ext
        print('Downloading:', final_img_links[index],
              'at', download_path)
        img = BytesIO(urllib.request.urlopen(final_img_links[index]).read())
        process_img(img, download_path)
    except urllib.error.URLError as e:
        print('Something went wrong while downloading:\n',
              final_img_links[index], e)
        n_errs += 1

print(f'{n_errs} images could not be downloaded, {n_imgs-n_errs} images successfully downloaded and processed.')
