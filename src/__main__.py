from dotenv import load_dotenv
import ai
import gee
import image


def main():
    load_dotenv()

    #ai.train_setup()
    #ai.load_generator_and_improve_image()

    gee.init_ee()    
    urls = gee.get_random_satellite_images()
    image.save_urls(urls)

    # image_url = urls[0] if urls else None
    # print('Thumbnail URL:', image_url)
    # if image_url:
    #     image.show_image(image_url)

if __name__ == '__main__':
    main()
