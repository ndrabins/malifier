import numpy
from scipy.misc import imsave



def extract_picture_from_bytes(filename):
    f = open(filename)

    #gets only the actual name of the
    filename_split = filename.split('/')
    filename = filename_split[-1][:-6] #strips out .bytes and gets only name

    file_text = f.readlines()
    num_lines = len(file_text)
    pixel_values = []
    #print(file_text[0:10])


    for i in range(0,num_lines):
        byte_line = file_text[i][9:-1].split(" ") #take out address locations, not important info

        #skip "??" garbage characters
        if "??" in byte_line:
            continue

        pixel_line = []
        for byte in byte_line:
            pixel_line.append(int(byte, 16)) #convert hex values to decimal pixel values
        pixel_values += pixel_line #append each line to full list

    '''
    dt = numpy.dtype('b')
    a = numpy.asarray(just_bytes[0:10])
    g = numpy.uint8(a)
    print(g)
    '''
    length = len(pixel_values)#num pixels
    width = int(length ** .5) #picture dimension
    rem = length % width #remove excess pixels for an even image
    pixel_array = numpy.asarray(pixel_values[:-rem])

    shaped_pixel_array = numpy.reshape(pixel_array, (int(length / width), width))
    imsave(filename + '.png', shaped_pixel_array)

    f.close()
    print("done")

#extract_picture_from_bytes('/media/ndrabins/My Passport/dataSample/0ACDbR5M3ZhBJajygTuf.bytes')