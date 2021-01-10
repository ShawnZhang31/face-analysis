import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
def putText(img, text, org, fontFace=None, fontScale=None, color=None, thickness=None, lineType=None, bottomLeftOrigin=None, fontPath=None, fontSize=32):
    """
    The enhanced version of  function cv::putText renders the specified text string in the image.

    @params
        img: Image.
        text: Text string to be drawn.
        org: Bottom-left corner of the text string in the image.
        fontFace: Font type.
        fontScale: Font scale factor that is multiplied by the font-specific base size.
        color: Text color.
        thickness: Thickness of the lines used to draw a text.
        lineType: Line type.
        bottomLeftOrigin: When true, the image data origin is at the bottom-left corner. Otherwise, . it is at the top-left corner.
        fontPath: The font file path.
    """
    if fontPath:
        font = ImageFont.truetype(fontPath, fontSize)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(org, text, font=font, fill=color)
        img = np.array(img_pil)
    else:
        cv2.putText(img, text, org, fontFace, fontScale, color, thickness=thickness, lineType=lineType, bottomLeftOrigin=bottomLeftOrigin)


