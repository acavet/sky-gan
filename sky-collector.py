'''Scrapes images from r/SkyPorn using Reddit API, formats for GAN training'''

import praw
import os
import urllib
from PIL import Image
from io import BytesIO
import config

# flashdrive to store downloaded pics on
imgs_path = '/Volumes/SKYFLASH/fromReddit/sky/'
allowed_extensions = ['.jpg', '.jpeg']
img_size = 32  # dimension of processed square image

reddit = praw.Reddit(client_id=config.client_id,
                     client_secret=config.client_secret,
                     user_agent='sky-collector')

subreddit = reddit.subreddit('SkyPorn')
posts = subreddit.hot(limit=1000)
img_urls = [post.url for post in posts]

downloaded_idx = 0  # to keep track of number valid downloaded imgs/add to img names


def process_img(img, img_path):
    # crops downloaded img to square, reduces size
    im = Image.open(img)
    w, h = im.size
    sq_len = min(w, h)
    sq_im = im.crop((0, 0, sq_len, sq_len))
    sq_small_im = sq_im.resize((img_size, img_size))
    sq_small_im.save(img_path)


# download and process jpg/jpeg imgs
for index, url in enumerate(img_urls):
    _, ext = os.path.splitext(url)
    if ext in allowed_extensions:
        try:
            download_path = imgs_path + \
                str(downloaded_idx) + ext
            print('downloading:', img_urls[index],
                  'at', download_path)
            img = BytesIO(urllib.request.urlopen(img_urls[index]).read())
            process_img(img, download_path)
            downloaded_idx += 1
        except urllib.error.URLError as e:
            print('something went wrong while downloading:\n',
                  img_urls[index], e)
