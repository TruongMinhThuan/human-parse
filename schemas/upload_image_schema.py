from pydantic import BaseModel

class UploadImageSchema(BaseModel):
    image_url: str or None = "https://nft-snap-dev.s3.ap-northeast-1.amazonaws.com/img-to-img/face_cropped_19000101-0000.png"
