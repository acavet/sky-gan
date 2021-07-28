'''Scrapes images from r/SkyPorn using Reddit API, formats for GAN training'''

import praw
import os
import urllib
from PIL import Image
import config

imgs_path = '/Volumes/skyflash/'  # flashdrive to store downloaded pics on
allowed_extensions = ['.jpg', '.jpeg']


reddit = praw.Reddit(client_id=config.client_id,
                     client_secret=config.client_secret,
                     user_agent='sky-collector')

subreddit = reddit.subreddit('SkyPorn')
posts = subreddit.hot(limit=5)
img_urls = [post.url for post in posts]

downloaded_idx = 0  # to keep track of number valid downloaded imgs/add to img names


def process_img(img_path):
    # crops downloaded img to square, reduces size to 256x256 pxls
    im = Image.open(img_path)
    w, h = im.size
    sq_len = min(w, h)
    sq_im = im.crop((0, 0, sq_len, sq_len))
    sq_small_im = sq_im.resize((256, 256))
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
            urllib.request.urlretrieve(
                img_urls[index], download_path)
            process_img(download_path)
            downloaded_idx += 1
        except urllib.error.URLError as e:
            print('something went wrong while downloading:\n',
                  img_urls[index], e)
