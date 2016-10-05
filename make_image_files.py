import glob
from extract_header_image import extract_picture_from_bytes

#this needs to be modified to whatever path you wanna use to store images.

path = "/media/ndrabins/My Passport/dataSample/"

#path = "/media/ndrabins/My Passport/train/"
extension = "*.bytes"

for fname in glob.glob(path+extension):
    print(fname)
    extract_picture_from_bytes(fname, path)