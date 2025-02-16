from skimage import io
import matplotlib.pyplot as plt
import os

def show_image(url):
    image = io.imread(url)
    plt.imshow(image)
    plt.show()

def save_urls(urls, filename='image-urls.txt'):
    # Read existing URLs from the file if it exists
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            existing_urls = set(file.read().splitlines())
    else:
        existing_urls = set()

    # Append new URLs without duplication
    with open(filename, 'a') as file:
        for url in urls:
            if url not in existing_urls:
                file.write(url + '\n')
                existing_urls.add(url)