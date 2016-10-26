import numpy
import scipy.misc
from scipy.misc import imsave

def extract_picture_from_bytes(filename, num_bytes=0):
    f = open(filename)

    #gets only the actual name of the file
    filename_split = filename.split('/')
    file_ext = filename[-5:]
    filename = filename_split[-1][:-6] #strips out .bytes and gets only name

    totBytes = 0

    pixel_values = []

    while totBytes < num_bytes:
        file_text = f.readline()
        if not file_text:
            break

        byte_line = file_text[9:-1].split(" ")  # take out address locations, not important info
        numBytes = 16

        if "??" in byte_line:
            continue

        for byte in byte_line:
            pixel_values.append(int(byte, 16)) #convert hex values to decimal pixel values

        totBytes += 16


    length = len(pixel_values)#num pixels
    width = int(length ** .5) #picture dimension
    if (width == 0):
        return []

    rem = length % (width ** 2) #remove excess pixels for an even image
    if (rem > 0):
        pixel_array = numpy.asarray(pixel_values[:-rem])
    else:
        pixel_array = numpy.asarray(pixel_values[:])

    shaped_pixel_array = pixel_array

    #this puts the array in the shape it will be for the CNN and then saves it as an image
    #using this I can open it and see what sort of images I will be getting

    #shaped_pixel_array2 = numpy.reshape(pixel_array, (width, width))
    #scipy.misc.imsave('/home/napster/outfile.png', shaped_pixel_array2)

    #instead of creating new files like below I am going to pass
    #imsave(filename + '.png', shaped_pixel_array)
    f.close()
    return shaped_pixel_array

#
#extract_picture_from_bytes('/media/ndrabins/My Passport/dataSample/0NyfGXt8nmlK72Q9Irhs.bytes')