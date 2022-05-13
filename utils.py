import base64
from PIL import Image
import io


def numpy_to_base64_String(image):
    pil_image = Image.fromarray(image)
    rawBytes = io.BytesIO()
    pil_image.save(rawBytes, "PNG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue())
    return img_base64.decode()