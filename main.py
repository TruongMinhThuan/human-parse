from typing import Union

from fastapi import FastAPI
from fastapi import FastAPI, File, Form, UploadFile
from fastapi import APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware

from schemas.upload_image_schema import UploadImageSchema
from services import image_service
import datetime
import os
from simple_extractor import gen_mask_scale, gen_mask_without_scale
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import requests

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.mount("/medias", StaticFiles(directory="medias"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/upload-image-transparent")
def upload_image(uploadImageRequest: UploadImageSchema, request: Request):

    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    upload_folder = f"medias/upload/{time_str}"

    # create folder if not exist
    if not os.path.exists(f"medias/upload/{time_str}"):
        os.makedirs(f"medias/upload/{time_str}")

    image_path = f"{upload_folder}/uploaded_image_{time_str}.jpg"

    local_path = image_service.save_image_url_to_file(
        uploadImageRequest.image_url, image_path)

    # Lip
    segment_path = gen_mask_wihout_scale(
        output_dir=upload_folder,
        input_dir=upload_folder,
    )

    # Atr
    # segment_path = gen_mask(
    #     output_dir=upload_folder,
    #     input_dir=upload_folder,
    #     datasets="atr",
    #     model_restore="checkpoints/atr.pth",
    # )

    print("segment_path", segment_path)
    domain = request.base_url
    # image_url = request.build_absolute_uri( f"/medias/{segment_path}")
    image_url = f"{domain}{segment_path}"

    print("segment_image_url", image_url)

    return {
        "message": "Image uploaded successfully",
        "image_path": local_path,
        "image_url": image_url
    }


@app.post("/upload-image-segmentation")
def upload_image_mask(uploadImageRequest: UploadImageSchema, request: Request):

    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    upload_folder = f"medias/upload/{time_str}"

    # create folder if not exist
    if not os.path.exists(f"medias/upload/{time_str}"):
        os.makedirs(f"medias/upload/{time_str}")

    image_path = f"{upload_folder}/uploaded_image_{time_str}.jpg"

    local_path = image_service.save_image_url_to_file(
        uploadImageRequest.image_url, image_path)



    # Lip
    segment_mask_scale_path = gen_mask_scale(
        output_dir=upload_folder,
        input_dir=upload_folder,
        mask_scale=11
    )
    print("segment_mask_scale_path", segment_mask_scale_path)

    domain = request.base_url
    # image_url = request.build_absolute_uri( f"/medias/{segment_path}")
    # transparent_image_url = f"{domain}{transparent_image_path}"
    segment_mask_scale_path_url = f"{domain}{segment_mask_scale_path}"
    

    return {
        "message": "Image uploaded successfully",
        "original_image_url": uploadImageRequest.image_url,
        # "transparent_image_url": transparent_image_url,
        "mask_image_url": segment_mask_scale_path_url
    }


@app.post("/upload-image-segmentation-transparent")
def upload_image_transparent(uploadImageRequest: UploadImageSchema, request: Request):

    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    upload_folder = f"medias/upload/{time_str}"

    # create folder if not exist
    if not os.path.exists(f"medias/upload/{time_str}"):
        os.makedirs(f"medias/upload/{time_str}")

    image_path = f"{upload_folder}/uploaded_image_{time_str}.jpg"

    local_path = image_service.save_image_url_to_file(
        uploadImageRequest.image_url, image_path)


    
    # Lip
    segment_mask_scale_path = gen_mask_scale(
        output_dir=upload_folder,
        input_dir=upload_folder,
        mask_scale=8
    )
    
 
    print("segment_mask_scale_path : ", segment_mask_scale_path)

    original_image_url = uploadImageRequest.image_url

    origin_image = Image.open(requests.get(original_image_url, stream=True).raw)

    overlay_image = Image.open(segment_mask_scale_path)

    origin_image.paste(overlay_image, (0, 0), overlay_image)
    
    transparent_image_path = f"{upload_folder}/uploaded_image_{time_str}.transparent.png"
    origin_image.save(transparent_image_path)


    #  convert white color to transparent
    img = Image.open(transparent_image_path)
    img = img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    img.save(transparent_image_path)

    
    
    print("segment_mask_scale_path", segment_mask_scale_path)

    domain = request.base_url
    # image_url = request.build_absolute_uri( f"/medias/{segment_path}")
    # transparent_image_url = f"{domain}{transparent_image_path}"
    transparent_image_path_url = f"{domain}{transparent_image_path}"
    

    return {
        "message": "Image uploaded successfully",
        "original_image_url": uploadImageRequest.image_url,
        "transparent_image_url": transparent_image_path_url,
    }


# @app.get("medias/{image_path}")
# def get_image(image_path: str):
#     return FileResponse(image_path)get_palette
