import validators
import base64
import requests
from PIL import Image, ImageDraw


def urlTobase64(url: str) -> str:
    try:
        # comment:
        base64_string = ""
        if validators.url(url):
            base64_bytes = base64.b64encode(requests.get(url).content)
            base64_string = base64_bytes.decode()
        else:
            base64_string = imageFileTobase64(url)

        return str(base64_string)
    except Exception as e:
        print("error base64: ", e)
        raise e
    # end try


def imageFileTobase64(file_path: str) -> str:
    try:
        # comment:
        with open(file_path, 'rb') as image_file:
            base64_bytes = base64.b64encode(image_file.read())

            base64_string = base64_bytes.decode()
            return base64_string
    except Exception as e:
        raise e
    # end try


def save_image_url_to_file(image_url: str, file_path: str) -> str:
    try:
        # comment:
        if validators.url(image_url):
            image = requests.get(image_url)
            with open(file_path, 'wb') as f:
                f.write(image.content)
        return file_path
    except Exception as e:
        raise e
    # end try


def base64ToImage(base64_string: str, file_path: str) -> str:
    try:
        # comment:
        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(base64_string))
        return file_path
    except Exception as e:
        raise e
    # end try


def get_image_size(img_path):
    with Image.open(img_path) as img:
        width, height = img.size
        return {
            "width": width,
            "height": height
        }


def resize_image_size(img_path, size=(512, 512)):
    with Image.open(img_path) as img:
        img = img.convert('RGB')

        img.resize(size).save(img_path)
        return img_path
