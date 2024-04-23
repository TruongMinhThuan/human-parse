from typing import Union

from fastapi import FastAPI
from fastapi import FastAPI, File, Form, UploadFile
from fastapi import APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware

from schemas.upload_image_schema import UploadImageSchema
from services import image_service
import datetime
import os
from simple_extractor import gen_mask
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


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


@app.post("/upload-image")
def upload_image(uploadImageRequest: UploadImageSchema, request: Request):

    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    upload_folder = f"medias/upload/{time_str}"

    # create folder if not exist
    if not os.path.exists(f"medias/upload/{time_str}"):
        os.makedirs(f"medias/upload/{time_str}")

    image_path = f"{upload_folder}/uploaded_image_{time_str}.jpg"

    local_path = image_service.save_image_url_to_file(
        uploadImageRequest.image_url, image_path)

    segment_path = gen_mask(
        output_dir=upload_folder,
        input_dir=upload_folder
    )

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


# @app.get("medias/{image_path}")
# def get_image(image_path: str):
#     return FileResponse(image_path)
