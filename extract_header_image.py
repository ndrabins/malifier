import numpy
from scipy.misc import imsave

def extract_picture_from_bytes(filename, write_path = "", num_bytes=0):
    f = open(filename)

    #gets only the actual name of the file
    filename_split = filename.split('/')
    file_ext = filename[-5:]
    filename = filename_split[-1][:-6] #strips out .bytes and gets only name
    file_text = f.readlines()

    if num_bytes != 0:
        #needs to be in increments of 16 because that is how many bytes are in a line
        num_bytes_to_add = num_bytes % 16
        num_lines = int(num_bytes+num_bytes_to_add/16)
    else:
        num_lines = len(file_text)
    pixel_values = []

    for i in range(0,num_lines):
        byte_line = file_text[i][9:-1].split(" ") #take out address locations, not important info

        #skip "??" garbage characters
        if "??" in byte_line:
            continue

        pixel_line = []
        for byte in byte_line:
            pixel_line.append(int(byte, 16)) #convert hex values to decimal pixel values
        pixel_values += pixel_line #append each line to full list


    length = len(pixel_values)#num pixels
    width = int(length ** .5) #picture dimension
    rem = length % width #remove excess pixels for an even image
    if rem == 0:
        pixel_array = numpy.asarray(pixel_values)
    else:
        pixel_array = numpy.asarray(pixel_values[:-rem])

    shaped_pixel_array = numpy.reshape(pixel_array, (int(length / width), width))
    imsave(write_path + filename + '_' + file_ext + '.png', shaped_pixel_array)
    #imsave(write_path + filename + '.png', shaped_pixel_array)

    f.close()
    print("done")

#extract_picture_from_bytes('/media/ndrabins/My Passport/dataSample/0NyfGXt8nmlK72Q9Irhs.bytes')