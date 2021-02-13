#!/usr/bin/env python
import argparse # to parse arguments like --help
from PIL import Image, ImageOps # pip install pillow - library to read picture files
import os.path # folder structure of operating system
import time # to implement timers of individual tasks
from math import cos, pi, sqrt # used for mathematical operations
from string import ascii_letters
# initialize parser
#  type= filetype
#  dest= variable
#  required= obvious
#  action="store_true"= makes a bool variable --verbose, --debug, -v, -d, -dv or -vd True in dest=
#  default= defines the default value if the argument is not given
parser = argparse.ArgumentParser(prog="jfake")
parser.add_argument("--input", "-i", required=True, help="Input image file [BMP], [GIF], [JPEG], [PNG], [PPM], [TIFF] (required)")
parser.add_argument("--output", "-o", type=str, default="output", help="Define output folder %(prog)s (default: %(default)s)")
parser.add_argument("--verbose", "-v", action="store_true", dest="verbose", help="Write all steps to terminal (default: False)")
parser.add_argument("--debug", "-d", action="store_true", dest="debug", help="Write all steps to output folder (default: False)")
parser.add_argument("--quality", "-q", type=int, default="50", dest="quality", help="JPEG-Quality [1-99] %(prog)s (default: %(default)s)")
parser.add_argument("--multiplier", "-m", type=int, default="0", dest="multiplier", help="Multiplier %(prog)s (default: Automatic)")
parser.add_argument("--benchmark", "-b", action="store_true", dest="benchmark", help="Write needed time per step in file (default: False)")
parser.add_argument("--entropy", "-e", action="store_true", dest="entropy", help="Calculate entropy in each processing step (default: False)")
parser.add_argument("--psnr", "-p", action="store_true", dest="psnr", help="Calculate signal-to-noise-ratio (PSNR) (default: False)")
parser.add_argument("--numba", "-n", action="store_true", dest="numba", help="Use numba jit compiler for better performance (default: False)")
parser.add_argument("--cupy", "-c", action="store_true", dest="cupy", help="Use cupy to allocate CUDA for better performance (default: False)")
args = parser.parse_args()

if args.numba:
    import numba as nb # pip install wheel - pip install numba

if args.cupy:
    import cupy as np # pip install cupy - used for mathematical operations using CUDA
else:
    import numpy as np # pip install numpy - used for mathematical operations

if args.quality <= 0 or args.quality > 99:
    raise Exception("The quality value should be in range of 1 to 99!")

if args.numba and args.cupy:
    raise Exception("Using Cupy and Numba at the same time is not possible.")

auto = False
if args.multiplier == 0:
    auto = True

# --input
infile = Image.open(args.input)
infile = infile.convert('RGB')
infilename = os.path.basename(args.input).rstrip(ascii_letters)[:-1]
inputpath = os.path.dirname(os.path.realpath(__file__))

# create outputfolder
if os.path.isdir(args.output)==False:
    os.mkdir(os.path.join(inputpath, args.output))

# Starting Benchmark
if args.benchmark:
    if os.path.isfile(os.path.join(inputpath, args.output, infilename + "_benchmark.txt")):
        os.remove(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"))
    benchtxt = open(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"), "a")
    benchstart = time.time()

# --debugging
# create debuggingfolder
if args.debug:
    if os.path.isdir(os.path.join(inputpath, args.output, "debug"))==False:
        os.mkdir(os.path.join(inputpath, args.output, "debug"))

# --verbose
if args.verbose:
    print(" jfake - parameters ".center(80,"*"))
    print("Input:       ", inputpath + "/" + args.input)
    print("Output:      ", os.path.join(inputpath, args.output, infilename + "_output.png"))
    print("Verbosity:   ", args.verbose)
    print("Debugging:   ", args.debug)
    print("JPEG-quality:", args.quality)
    if auto:
        print("Multiplier:   Automatic")
    else:
        print("Multiplier:  ", args.multiplier)
    print("Entropy:     ", args.entropy)
    print("PSNR:        ", args.psnr)
    print("Benchmark:   ", args.benchmark)
    print(" jfake - starting computing ".center(80,"*"))
if os.path.isfile(os.path.join(inputpath, args.output, infilename + "_arguments.txt")):
    os.remove(os.path.join(inputpath, args.output, infilename + "_arguments.txt"))
with open(os.path.join(inputpath, args.output, infilename + "_arguments.txt"), "w") as arguments:
    print(" jfake - parameters ".center(80,"*"), file=arguments)
    print("Input:       ", inputpath + "/" + args.input, file=arguments)
    print("Output:      ", os.path.join(inputpath, args.output, infilename + "_output.png"), file=arguments)
    print("Verbosity:   ", args.verbose, file=arguments)
    print("Debugging:   ", args.debug, file=arguments)
    print("JPEG-quality:", args.quality, file=arguments)
    if auto:
        print("Multiplier:   Automatic", file=arguments)
    else:
        print("Multiplier:  ", args.multiplier, file=arguments)
    print("Entropy:     ", args.entropy, file=arguments)
    print("PSNR:        ", args.psnr, file=arguments)
    print("Benchmark:   ", args.benchmark, file=arguments)

class Trafo:
    """manages the colorspace transformation rgb -> ycbcr including the splitting into 8x8 blocks"""
    def __init__(self, infile, infilename):
        self.pic_name = args.input
        self.img = infile

        #get width and height of img
        self.width, self.height = self.img.size
        self.width_e = self.height_e = None
        self.img_ext = self.__extend_img()
        self.r = np.array(self.img_ext.getchannel('R')).flatten()
        self.g = np.array(self.img_ext.getchannel('G')).flatten()
        self.b = np.array(self.img_ext.getchannel('B')).flatten()
        self.rgb = np.concatenate((np.array([self.r]), np.array([self.g]), np.array([self.b])))
        self.trafo_table = np.array([
        [0.299,0.587,0.114],
        [-0.169,-0.331,0.5],
        [0.5,-0.419,-0.081]]).astype(np.float32)
        self.backtrafo_table = np.array([
        [1, 0, 1.402],
        [1, -0.34414, -0.71414],
        [1, 1.772, 0]]).astype(np.float32)

    def __extend_img(self):
        """create new pics and extend if necessary, so it's dividable by 8"""

        #pixel to add, so it is dividable by 8
        if (self.width % 8) != 0:
            width_diff = 8 - (self.width % 8)
        else:
            width_diff = 0
        if (self.height % 8) != 0:
            height_diff = 8 - (self.height % 8)
        else:
            height_diff = 0

        #create new image with additional pixels if width or height is not dividable by 8
        self.width_e = self.width + width_diff
        self.height_e = self.height + height_diff

        #if width and height are already dividable by 8, no extension needed
        if (width_diff == 0) and (height_diff == 0):
            return self.img

        lr_mir = ImageOps.mirror(self.img)
        tb_mir = ImageOps.flip(self.img)
        corner_mir = ImageOps.mirror(tb_mir)
        img_extended = self.create_extenisons(lr_mir, tb_mir, corner_mir, width_diff, height_diff)

        if(args.debug):
            img_extended.save(os.path.join(inputpath, args.output, infilename + "_img_extended.png"))
        return img_extended

    def create_extenisons(self, lr_mir, tb_mir, corner_mir, width_diff, height_diff):
        """mirroring and flipping image at the sides and corners and cropping the centre one for usage"""
        img_tmp = Image.new('RGB', (3*self.width, 3*self.height))
        img_tmp.paste(corner_mir, (0,0))
        img_tmp.paste(tb_mir, (self.width, 0))
        img_tmp.paste(corner_mir, (2*self.width, 0))
        img_tmp.paste(lr_mir, (0, self.height))
        img_tmp.paste(self.img, (self.width, self.height))
        img_tmp.paste(lr_mir, (2*self.width, self.height))
        img_tmp.paste(corner_mir, (0, 2*self.height))
        img_tmp.paste(tb_mir, (self.width, 2*self.height))
        img_tmp.paste(corner_mir, (2*self.width, 2*self.height))
        left_upper_x = self.width - (width_diff // 2)
        left_upper_y = self.height - (height_diff // 2)
        right_lower_x = left_upper_x + self.width_e
        right_lower_y = left_upper_y + self.height_e
        img_extended = img_tmp.crop((left_upper_x, left_upper_y, right_lower_x, right_lower_y))
        return img_extended

    def __write_to_txtfile(self, y8x8, cb8x8, cr8x8):
        """writing the values 8x8 blocks of each pic (y, cb, cr) to a textfile"""
        with open(os.path.join(inputpath, args.output, "debug", infilename + "_y.txt"), "w") as y_txt:
            with open(os.path.join(inputpath, args.output, "debug", infilename + "_cb.txt"), "w") as cb_txt:
                with open(os.path.join(inputpath, args.output, "debug", infilename + "_cr.txt"), "w") as cr_txt:
                    for i, block in enumerate(y8x8):
                        y_txt.write("Block: " + str(i+1) + "\n")
                        cb_txt.write("Block: " + str(i+1) + "\n")
                        cr_txt.write("Block: " + str(i+1) + "\n")
                        for j in range(8):
                            for k in range(8):
                                y_txt.write(str(j*8+k+1) + ": " + str(y8x8[i][j][k]) + "; ")
                                cb_txt.write(str(j*8+k+1) + ": " + str(cb8x8[i][j][k]) + "; ")
                                cr_txt.write(str(j*8+k+1) + ": " + str(cr8x8[i][j][k]) + "; ")
                        y_txt.write("\n")
                        cb_txt.write("\n")
                        cr_txt.write("\n")

    def rgb2ycbcr(self):
        """main colorspace transformation from r,g,b to y,cb,cr
            input:
            return: y, cb, cr values as lists containing data for one image (y,cb,cr)"""
        if(args.verbose):
            print("Performing colorspace transformation from RGB to YCbCr")

        if self.img.mode != 'RGB' and self.img.mode != 'YCbCr' and self.img.mode != 'RGBA':
            raise Exception("The Picture has mode: " + self.img.mode + " but it must be RGB!")
        if args.numba:
            y, cb, cr = numba_rgb2ycbcr(self.trafo_table, self.rgb)
        else:
            ycbcr = self.trafo_table @ self.rgb
            y = ycbcr[0]
            cb = ycbcr[1] + 128
            cr = ycbcr [2] + 128

            y =  np.rint(y).astype(np.uint8)
            cb = np.rint(cb).astype(np.uint8)
            cr = np.rint(cr).astype(np.uint8)

        if(args.debug):
            self.save_ycbcr_channels(y, cb, cr)
        if args.entropy:
            with open(os.path.join(inputpath, args.output, infilename + "_entropy.txt"), "w") as entropy_txt:
                entropy_txt.write("Entropy before jpeg encoding".center(48,"*") + "\n")
            print("- Entropy before jpeg encoding")
            Entropy.execute_entropy(self, self.r, self.g, self.b, rgb=True, pixels=True)
            Entropy.execute_entropy(self, y, cb, cr, rgb=False, pixels=True)

        return y, cb, cr

    def save_ycbcr_channels(self, y, cb, cr):
        """writes each channel in a png file"""
        y = y.reshape(self.height_e, self.width_e)
        cb = cb.reshape(self.height_e, self.width_e)
        cr = cr.reshape(self.height_e, self.width_e)
        imgy = Image.fromarray(y, 'L')
        imgcb = Image.fromarray(cb, 'L')
        imgcr = Image.fromarray(cr, 'L')
        imgy.save(os.path.join(inputpath, args.output, "debug", infilename + "_y.png"))
        imgcb.save(os.path.join(inputpath, args.output, "debug", infilename + "_cb.png"))
        imgcr.save(os.path.join(inputpath, args.output, "debug", infilename + "_cr.png"))

    def split8x8(self, y, cb, cr, h, w):
        """splits into 8x8 blocks"""
        y8x8 = y.reshape(h//8, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8)
        cb8x8 = cb.reshape(h//8, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8)
        cr8x8 = cr.reshape(h//8, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8)

        if args.debug:
            self.__write_to_txtfile(y8x8, cb8x8, cr8x8)
        return y8x8, cb8x8, cr8x8

    def ycbcr2rgb(self, y=None, cb=None, cr=None):
        """colorspacetransformation from y,cb,cr to rgb
            input: y,cb,cr values saved as lists
            channels = true for rgb arrays, false for pil image
            return: rgb image as PIL.Image"""
        if(args.verbose):
            print("Performing colorspace transformation from YCbCr to RGB")

        ycbcr = np.array((y,cb,cr))
        if args.numba:
            rgb = numba_ycbcr2rgb(self.backtrafo_table, ycbcr)
        else:
            ycbcr[1] -= 128
            ycbcr[2] -= 128
            rgb = self.backtrafo_table @ ycbcr
            rgb = np.rint(rgb).astype(np.int16)

        rgb = rgb.reshape(3, self.height_e, self.width_e)
        rgb = np.clip(rgb, 0, 255)
        # bring axes in correct order
        rgb = np.swapaxes(rgb, 0,2)
        rgb = np.swapaxes(rgb, 0,1)
        if args.cupy:
            rgb_img = Image.fromarray(np.asnumpy(rgb).astype(np.uint8),'RGB')
        
        else:
            rgb_img = Image.fromarray(rgb.astype(np.uint8),'RGB')

        crop_recomb = self.cropping_back(rgb_img)

        r = np.array(crop_recomb.getchannel('R')).flatten().astype(np.int16)
        g = np.array(crop_recomb.getchannel('G')).flatten().astype(np.int16)
        b = np.array(crop_recomb.getchannel('B')).flatten().astype(np.int16)
        rgb_out = np.concatenate((np.array([r]), np.array([g]), np.array([b])))

        if(args.debug):
            crop_recomb.save(os.path.join(inputpath, args.output, "debug", infilename + "_rücktrafo.png"))

        return crop_recomb, rgb_out

    def cropping_back(self, rgb_img):
        width_diff = self.width_e - self.width
        height_diff = self.height_e - self.height
        left_upper_x = width_diff // 2
        left_upper_y = height_diff // 2
        right_lower_x = left_upper_x + self.width
        right_lower_y = left_upper_y + self.height
        crop_recomb = rgb_img.crop((left_upper_x, left_upper_y, right_lower_x, right_lower_y)) 
        return crop_recomb       

    def get_rgb(self):
        r = np.array(self.img.getchannel('R')).flatten()
        g = np.array(self.img.getchannel('G')).flatten()
        b = np.array(self.img.getchannel('B')).flatten()
        rgb = np.concatenate((np.array([r]), np.array([g]), np.array([b])))
        return rgb.astype(np.int16)

    def get_size(self):
        return self.height, self.width

    def recombine8x8(self, y8x8=None, cb8x8=None, cr8x8=None):
        """recombine the 8x8 blocks of y, cb and cr back to one picture each
            input: y,cb,cr values saved as list of lists, each list containing 8x8 blocks for one image (y,cb,cr)
            return: recombined images saved as lists containing data for one image (y,cb,cr)"""
        #if None in (y8x8, cb8x8, cr8x8):
         #   raise Exception('No valid values in y8x8, cb8x8, cr8x8!')
        y = y8x8.reshape(self.height_e//8, -1, 8, 8).swapaxes(1,2)
        cb = cb8x8.reshape(self.height_e//8, -1, 8, 8).swapaxes(1,2)
        cr = cr8x8.reshape(self.height_e//8, -1, 8, 8).swapaxes(1,2)
        return y.flatten(), cb.flatten(), cr.flatten()

class DCT:
    """manages FDCT"""
    def __init__(self, b8x8, infile, infilename):
        self.pic_name = args.input
        self.b8x8 = b8x8
        self.y8x8 = self.b8x8[0]
        self.cb8x8 = self.b8x8[1]
        self.cr8x8 = self.b8x8[2]

    def __write_to_txtfileFDCT(self, y8x8, cb8x8, cr8x8):
        """writing the values 8x8 blocks of each FDCT-pic (y, cb, cr) to a textfile"""
        y8x8 = np.round(y8x8, 3)
        cb8x8 = np.round(cb8x8, 3)
        cr8x8 = np.round(cr8x8, 3)
        with open(os.path.join(inputpath, args.output, "debug", infilename + "_FDCT_y.txt"), "w") as FDCT_y_txt:
            with open(os.path.join(inputpath, args.output, "debug", infilename + "_FDCT_cb.txt"), "w") as FDCT_cb_txt:
                with open(os.path.join(inputpath, args.output, "debug", infilename + "_FDCT_cr.txt"), "w") as FDCT_cr_txt:
                    for i, block in enumerate(y8x8):
                        FDCT_y_txt.write("Block: " + str(i+1) + "\n")
                        FDCT_cb_txt.write("Block: " + str(i+1) + "\n")
                        FDCT_cr_txt.write("Block: " + str(i+1) + "\n")
                        for j in range(64):
                            FDCT_y_txt.write(str(j+1) + ": " + str(y8x8[i][j]) + "; ")
                            FDCT_cb_txt.write(str(j+1) + ": " + str(cb8x8[i][j]) + "; ")
                            FDCT_cr_txt.write(str(j+1) + ": " + str(cr8x8[i][j]) + "; ")
                        FDCT_y_txt.write("\n")
                        FDCT_cb_txt.write("\n")
                        FDCT_cr_txt.write("\n")

    def __write_to_txtfileIDCT(self, y8x8, cb8x8, cr8x8):
        """writing the values 8x8 blocks of each IDCT-pic (y, cb, cr) to a textfile"""
        y8x8 = np.round(y8x8, 1)
        cb8x8 = np.round(cb8x8, 1)
        cr8x8 = np.round(cr8x8, 1)
        with open(os.path.join(inputpath, args.output, "debug", infilename + "_IDCT_y.txt"), "w") as IDCT_y_txt:
            with open(os.path.join(inputpath, args.output, "debug", infilename + "_IDCT_cb.txt"), "w") as IDCT_cb_txt:
                with open(os.path.join(inputpath, args.output, "debug", infilename + "_IDCT_cr.txt"), "w") as IDCT_cr_txt:
                    for i, block in enumerate(y8x8):
                        IDCT_y_txt.write("Block: " + str(i+1) + "\n")
                        IDCT_cb_txt.write("Block: " + str(i+1) + "\n")
                        IDCT_cr_txt.write("Block: " + str(i+1) + "\n")
                        for j in range(64):
                            IDCT_y_txt.write(str(j+1) + ": " + str(y8x8[i][j]) + "; ")
                            IDCT_cb_txt.write(str(j+1) + ": " + str(cb8x8[i][j]) + "; ")
                            IDCT_cr_txt.write(str(j+1) + ": " + str(cr8x8[i][j]) + "; ")
                        IDCT_y_txt.write("\n")
                        IDCT_cb_txt.write("\n")
                        IDCT_cr_txt.write("\n")

    def __compute_dct_table(self):
        """Computes each component of the transformation matrix for the dct. Returns a 8x8 ndarray """
        dct_table = np.empty((8,8))
        for i in range(8):
            for j in range(8):
                if i == 0:
                    c = 1/sqrt(2)
                else:
                    c = 1
                dct_table[i, j] = sqrt(2/8) * c * cos((pi*i*(2*j+1))/(2*8))
        return np.array(dct_table.astype(np.float32))

    def compute_DCT(self, f8x8, forward):
        """splits lists with 64 elements into 8x8 element lists and 
        returns list of 8x8 blocks with DCT coefficients. Forward = True for FDCT, False for IDCT"""
        dct_table = self.__compute_dct_table()
        if args.numba:
            #f8x8 = f8x8.astype(np.float64)
            f8x8 = numba_DCT(f8x8, dct_table, forward)
        else:
            if forward:
                # G: DCT spectrum, g: Input signal A: transformation matrix
                # FDCT(g) = G = A * g * A^T
                f8x8 = dct_table @ f8x8 @ dct_table.T
            else:
                # IDCT(G) = g = A^T * G * A
                f8x8 = dct_table.T @ f8x8 @ dct_table
        return f8x8

    def execute_DCT(self, forward):
        """Executes FDCT-computation for each color space. 
        Forward = True for FDCT, False for IDCT"""        
        if forward==True:
            if(args.verbose):
                print("Computing FDCT")
            yDCT = self.compute_DCT(self.y8x8, forward)
            cbDCT = self.compute_DCT(self.cb8x8, forward)
            crDCT = self.compute_DCT(self.cr8x8, forward)

            if args.entropy:
                with open(os.path.join(inputpath, args.output, infilename + "_entropy.txt"), "a") as entropy_txt:
                    entropy_txt.write("Entropy after FDCT".center(48,"*") +"\n")
                print("- Entropy after FDCT but before Quantization")
                Entropy.execute_entropy(self, yDCT,cbDCT,crDCT, rgb=False, pixels=False)
        else :
            """Executes IDCT-computation for each color space"""
            if(args.verbose):
                print("Computing IDCT")
            yDCT = self.compute_DCT(self.y8x8, forward)
            cbDCT = self.compute_DCT(self.cb8x8, forward)
            crDCT = self.compute_DCT(self.cr8x8, forward)
        if args.debug:
            self.reshape(yDCT, cbDCT, crDCT, forward)
        return yDCT, cbDCT, crDCT

    def reshape(self, y, cb, cr, forward):
        """Executes merging of blocks and writing of IDCT textfiles"""
        yO = y.reshape(-1, 64)
        cbO = cb.reshape(-1, 64)
        crO = cr.reshape(-1, 64)
        if forward:
            print("Writing FDCT.txt")
            self.__write_to_txtfileFDCT(yO, cbO, crO)
        else:
            print("Writing IDCT.txt")
            self.__write_to_txtfileIDCT(yO, cbO, crO)

class Quantization:
    """manages Quantization with given Quality-value and Dequantization"""

    def __init__(self, imgs8x8, quality, infilename):
        self.quality = quality
        self.y8x8 = imgs8x8[0]
        self.cb8x8 = imgs8x8[1]
        self.cr8x8 = imgs8x8[2]

        # quantization table of luminances
        self.qbase_y = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]]).astype(np.int8)

        # quantization table of chrominance
        self.qbase_c = np.array([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]]).astype(np.int8)

        self.quality_qtable_y = self.__compute_qual_qtable(self.qbase_y)
        self.quality_qtable_c = self.__compute_qual_qtable(self.qbase_c)

        if(args.debug):
            self.__write_qtables_to_textfile(self.quality_qtable_y, self.quality_qtable_c)

        tmp = self.__check_qualityvalue(self.quality_qtable_y, self.quality_qtable_c)
        Q = np.rint(tmp)

        if(args.verbose):
            print("approximated computed check for quality value: " + str(Q))
            print(" jfake - beginning backtransformation ".center(80,"*"))

    def __write_to_txtfile(self, y8x8, cb8x8, cr8x8):
        """writing the values 8x8 blocks of each pic (y, cb, cr) to a textfile"""
        y8x8 = y8x8.reshape(y8x8.shape[0], 64)
        cb8x8 = cb8x8.reshape(cb8x8.shape[0], 64)
        cr8x8 = cr8x8.reshape(cr8x8.shape[0], 64)

        with open(os.path.join(inputpath, args.output, "debug", infilename + "_quantization_y.txt"), "w") as y_txt:
            with open(os.path.join(inputpath, args.output, "debug", infilename + "_quantization_cb.txt"), "w") as cb_txt:
                with open(os.path.join(inputpath, args.output, "debug", infilename + "_quantization_cr.txt"), "w") as cr_txt:
                    for i, block in enumerate(y8x8):
                        y_txt.write("Block: " + str(i+1) + "\n")
                        cb_txt.write("Block: " + str(i+1) + "\n")
                        cr_txt.write("Block: " + str(i+1) + "\n")
                        for j in range(64):
                            y_txt.write(str(j+1) + ": " + str(y8x8[i,j]) + "; ")
                            cb_txt.write(str(j+1) + ": " + str(cb8x8[i,j]) + "; ")
                            cr_txt.write(str(j+1) + ": " + str(cr8x8[i,j]) + "; ")
                        y_txt.write("\n")
                        cb_txt.write("\n")
                        cr_txt.write("\n")

    def __write_qtables_to_textfile(self, qtable_y, qtable_c):
        """Writes computed tables in a textfile"""
        with open(os.path.join(inputpath, args.output, "debug", infilename + "_Quantizationtable_y.txt"), "w") as qtable_y_txt:
            with open(os.path.join(inputpath, args.output, "debug", infilename + "_Quantizationtable_c.txt"), "w") as qtable_c_txt:
                for i in range (8):
                    qtable_y_txt.write(str(qtable_y[i,:]) + "\n")
                    qtable_c_txt.write(str(qtable_c[i,:]) + "\n")

    def __check_qualityvalue(self, qtable_y, qtable_c):
        """calculates the approximate quality value from the computed qunatization tables"""
        # calculate mean of all entrys and subtract 1/64 * first entry to exclude
        avg_y = np.mean(qtable_y) - qtable_y[0,0]/64
        avg_c = np.mean(qtable_c) - qtable_c[0,0]/64
        m = (avg_y + 2*avg_c) / 3
        D = (abs(avg_y-avg_c) * 0.49) * 2
        Q = 100 - m + D
        return Q.astype(np.float32)

    def __compute_qual_qtable(self, table):
        """computes new quantization table from quality value and base quantization table"""
        if self.quality < 50:
            S = 5000//self.quality
        else:
            S = 200 - 2*self.quality
        quality_qtable = (np.array([S])*table + 50) / 100
        quality_qtable = np.floor(quality_qtable).astype(np.uint8)
        return quality_qtable

    def __quantize(self, img8x8, quality_table, forward):
        """divides dct coefficients with quantization matrix"""
        quality_table = quality_table.astype(np.float64)
        if args.numba:
            img8x8 = numba_quantize(img8x8, quality_table, forward)
        else:
            if forward:
                img8x8 /= quality_table
                np.rint(img8x8, img8x8)
            else:
                img8x8 *= quality_table
        return img8x8

    def quantize(self, y_q, cb_q, cr_q, forward):
        """Executes Quantization"""
        if forward:
            y = self.__quantize(self.y8x8, self.quality_qtable_y, forward)
            cb = self.__quantize(self.cb8x8, self.quality_qtable_c, forward)
            cr = self.__quantize(self.cr8x8, self.quality_qtable_c, forward)
            if(args.debug):
                self.__write_to_txtfile(y, cb, cr)
            if args.entropy:
                with open(os.path.join(inputpath, args.output, infilename + "_entropy.txt"), "a") as entropy_txt:
                    entropy_txt.write("Entropy after quantization".center(48,"*") + "\n")
                print("- Entropy after Quantization")
                Entropy.execute_entropy(self, y, cb, cr, rgb=False, pixels=False)
        else:
            if self.quality_qtable_y.size == 0 or self.quality_qtable_c.size == None:
                raise Exception("Please execute qunantization before dequantization!")
            y = self.__quantize(y_q, self.quality_qtable_y, forward)
            cb = self.__quantize(cb_q, self.quality_qtable_c, forward)
            cr = self.__quantize(cr_q, self.quality_qtable_c, forward)
        return y, cb, cr

class Entropy:
    """contains functions for calculating and printing entropy values of images"""
    def execute_entropy(self, c1,c2,c3, rgb, pixels):
        """Takes list of lists of 64 elements as input.
        Computes the mean information H per symbol and writes it to entropy.txt.
        rgb = true for rgb image, false for ycbcr image
        pixels = true for pixelvalues, false for DCT coefficients"""

        if rgb == True:
            with open (os.path.join(inputpath, args.output, infilename + "_entropy.txt"), "a") as entropy_txt:
                for i, channel in enumerate([c1, c2, c3]):
                    H = Entropy.__compute_entropy(self, channel)
                    if i == 0:
                        channel_string = "R "
                    elif i == 1:
                        channel_string = "G "
                    elif i == 2:
                        channel_string = "B "
                    print("- Entropy in channel", str(channel_string) + ":", str(H), "bit per pixel")
                    entropy_txt.write("Entropy in channel " + str(channel_string) + " = " + str(H) + " bit per pixel" + "\n")
        else:
            with open (os.path.join(inputpath, args.output, infilename + "_entropy.txt"), "a") as entropy_txt:
                for i, channel in enumerate([c1, c2, c3]):
                    channel = np.rint(channel.flatten()).astype(np.int16)
                    H = Entropy.__compute_entropy(self, channel)
                    if i == 0:
                        channel_string = "Y "
                    elif i == 1:
                        channel_string = "Cb"
                    elif i == 2:
                        channel_string = "Cr"
                    if pixels == True:
                        print("- Entropy in channel", str(channel_string) + ":", str(H), "bit per pixel")
                        entropy_txt.write("Entropy in channel " + str(channel_string) + " = " + str(H) + " bit per pixel" + "\n")
                    else:
                        print("- Entropy in channel", str(channel_string) + ":", str(H), "bit per coefficient")
                        entropy_txt.write("Entropy in channel " + str(channel_string) + " = " + str(H) + " bit per coefficient" + "\n")

    def __compute_entropy(self, values):
        from collections import Counter
        """Builds a dictionary of values with total probabilities and returns entropy H
        H = -sum(p(i)*log(p(i)))  bit/symbol"""
        if args.cupy:
            values = np.asnumpy(values)

        value_counter = Counter(values)
        H = 0
        pixel = len(values)
        for value in value_counter.values():
            pi = value/pixel
            H = H + pi*np.log2(pi)
        H = -round(float(H), 3)
        return H

if args.numba:
    @nb.njit(parallel=True, cache=True)
    def numba_rgb2ycbcr(trafo_table, rgb):
        rgb = rgb.astype(nb.float32)
        ycbcr = trafo_table @ rgb
        np.rint(ycbcr, ycbcr)
        y = ycbcr[0]
        cb = (ycbcr[1] + 128)
        cr = (ycbcr[2] + 128)
        return y, cb, cr

    @nb.njit(parallel=True, cache=True)
    def numba_ycbcr2rgb(trafo_table, ycbcr):
        ycbcr[1] -= 128
        ycbcr[2] -= 128
        rgb = trafo_table @ ycbcr
        np.rint(rgb, rgb)
        rgb = rgb.astype(nb.int16)
        return rgb

    @nb.njit(parallel=True, cache=True)
    def numba_DCT(f8x8, dct_table, forward):
        f8x8 = f8x8.astype(nb.float32)
        dct = np.empty_like(f8x8)
        if forward:
            # G: DCT spectrum, g: Input signal A: transformation matrix
            # FDCT(g) = G = A * g * A^T
            for i in nb.prange(dct.shape[0]):
                dct[i] = dct_table @ f8x8[i] @ dct_table.T
        else:
            # IDCT(g) = g = A^T * G * A
            for i in nb.prange(dct.shape[0]):
                dct[i] = dct_table.T @ f8x8[i] @ dct_table
        return dct

    @nb.njit(parallel=True, cache=True, error_model='numpy', fastmath=True)
    #@nb.njit(parallel=True, cache=True)

    def numba_quantize(img8x8, quality_table, forward):
        if forward:
            for i in nb.prange(img8x8.shape[0]):
                img8x8[i] /= quality_table
            np.rint(img8x8, img8x8)
        else:
            for i in nb.prange(img8x8.shape[0]):
                img8x8[i] *= quality_table
        return img8x8

def compute_psnr(diff_img):
    """takes diff Image as inout and returns PSNR and Minimum SNR for each color channel"""
    diff = np.asarray(diff_img)
    diff = diff.reshape(3, diff_img.width * diff_img.height)
    mse_rgb = np.mean(diff**2, axis = 1)
    psnr_rgb = 10 * np.log10(255**2 / mse_rgb)
    min_mse_rgb = np.amax(diff, axis = 1)
    min_snr_rgb = 10 * np.log10(255**2 / min_mse_rgb)
    return  psnr_rgb[0], psnr_rgb[1], psnr_rgb[2], min_snr_rgb[0], min_snr_rgb[1], min_snr_rgb[2]

def subtract_images(img_input, img_output, img_height, img_width):
    """Subtract input and output image and compute difference image and ela image"""
    input_data = img_input.T
    output_data = img_output.T
    if len(input_data) != len(output_data):
        raise Exception("Input and Output image have different sizes!")

    diff = abs(input_data - output_data)
    diff = diff.reshape(img_height, img_width, 3)
    diff = np.clip(diff, 0, 255)

    if auto:
        args.multiplier = np.divide(255, diff.max())

    diff_multiplied = diff * args.multiplier
    diff_multiplied = np.clip(diff_multiplied, 0, 255)
    
    if args.cupy:
        diff_img = Image.fromarray(np.asnumpy(diff).astype(np.uint8), 'RGB')
        diff_img_multiplied = Image.fromarray(np.asnumpy(diff_multiplied).astype(np.uint8), 'RGB')
    else:
        diff_img = Image.fromarray(diff.astype(np.uint8), 'RGB')
        diff_img_multiplied = Image.fromarray(diff_multiplied.astype(np.uint8), 'RGB')
    return diff_img, diff_img_multiplied

def trafo_dct_q():
    """Main function"""
    "rgb -> ycbcr transformation"
    start = time.time()
    tr = Trafo(infile, infilename)
    rgb_input = tr.get_rgb()
    y, cb, cr = tr.rgb2ycbcr()
    fdct = DCT(tr.split8x8(y, cb, cr, tr.height_e, tr.width_e), infile, infilename)
    end = time.time()
    print("* Time elapsed for RGB -> YCbCr:       "+ format((end - start), '.3f') + 's')
    if args.benchmark:
        with open(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"), 'a') as file:
            file.write("   jfake Benchmark\nRGB -> YCbCr: "+ format((end - start), '.3f') + 's\n')

    """fdct"""
    start = time.time()
    FDCT64 = fdct.execute_DCT(forward=True)
    end = time.time()
    print("* Time elapsed for FDCT:               "+ format((end - start), '.3f') + 's')
    if args.benchmark:
        with open(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"), 'a') as file:
            file.write("FDCT:         "+ format((end - start), '.3f') + 's\n')

    """quantization"""
    start = time.time()
    qztion = Quantization(FDCT64, args.quality, infilename)
    y, cb, cr = qztion.quantize(None, None, None, forward=True)
    end = time.time()
    print("* Time elapsed for quantization:       " + format((end - start), '.3f') + 's')
    ycbcr = qztion.quantize(y, cb, cr, forward=False)
    end2 = time.time()
    print("* Time elapsed for dequantization:     " + format((end2 - end), '.3f') + 's')
    if args.benchmark:
        with open(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"), 'a') as file:
            file.write("Quantization: " + format((end - start), '.3f') + 's\n')

    """IDCT"""
    start = time.time()
    idct = DCT(ycbcr, infile, infilename)
    IDCT64 = idct.execute_DCT(forward=False)
    end = time.time()
    print("* Time elapsed for IDCT:               "+ format((end - start), '.3f') + 's')
    if args.benchmark:
        with open(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"), 'a') as file:
            file.write("IDCT:         "+ format((end - start), '.3f') + 's\n')

    if(args.verbose):
        print("Performing colorspace transformation from YCbCr to RGB")
    start = time.time()
    y_r,cb_r,cr_r = tr.recombine8x8(IDCT64[0], IDCT64[1], IDCT64[2])
    out_img, rgb_output = tr.ycbcr2rgb(y_r,cb_r,cr_r)
    end = time.time()
    print("* Time elapsed for YCbCr -> RGB:       " + format((end - start), '.3f') + 's')
    start_subtraction = time.time()   
    img_height, img_width = tr.get_size()
    diff_img, diff_img_multiplied = subtract_images(rgb_input, rgb_output, img_height, img_width)
    if(args.debug):
        diff_img.save(os.path.join(inputpath, args.output, "debug", infilename + "_diff_img_original.png"))
    if args.psnr:
        noise_r, noise_g, noise_b, min_noise_r, min_noise_g, min_noise_b = compute_psnr(diff_img)
        if os.path.isfile(os.path.join(inputpath, args.output, infilename + "_psnr.txt")):
            os.remove(os.path.join(inputpath, args.output, infilename + "_psnr.txt"))
        print("- PSNR R:", str(round(noise_r, 3)), "dB")
        print("- PSNR G:", str(round(noise_g, 3)), "dB")
        print("- PSNR B:", str(round(noise_b, 3)), "dB")
        print("- Min SNR R:", str(round(min_noise_r, 3)), "dB")
        print("- Min SNR G:", str(round(min_noise_g, 3)), "dB")
        print("- Min SNR B:", str(round(min_noise_b, 3)), "dB")
        with open(os.path.join(inputpath, args.output, infilename + "_psnr.txt"), 'w') as snr_txt:
            print("PSNR R:", str(round(noise_r, 3)), "dB", file=snr_txt)
            print("PSNR G:", str(round(noise_g, 3)), "dB", file=snr_txt)
            print("PSNR B:", str(round(noise_b, 3)), "dB", file=snr_txt)
            print("Min SNR R:", str(round(min_noise_r, 3)), "dB", file=snr_txt)
            print("Min SNR G:", str(round(min_noise_g, 3)), "dB", file=snr_txt)
            print("Min SNR B:", str(round(min_noise_b, 3)), "dB", file=snr_txt)

    end_subtraction = time.time()
    print("* Time elapsed for subtracting images: " + format((end_subtraction - start_subtraction), '.3f') + 's')
    if args.benchmark:
        with open(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"), 'a') as file:
            file.write("YCbCr -> RGB: "+ format((end - start), '.3f') + 's\n')
    return out_img, diff_img_multiplied

out_img, diff_img_multiplied = trafo_dct_q()

# Writing output.png
start = time.time()
out_img.save(os.path.join(inputpath, args.output, infilename + "_output.png"))
diff_img_multiplied.save(os.path.join(inputpath, args.output, infilename + "_diff.png"))
end = time.time()
print("* Time elapsed for writing file:       " + format((end - start), '.3f') + 's')
if args.benchmark:
        with open(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"), 'a') as file:
            file.write("Writing file: "+ format((end - start), '.3f') + 's\n')

# Write used arguments in .txt file
if args.benchmark:
    benchstop = time.time()
    with open(os.path.join(inputpath, args.output, infilename + "_benchmark.txt"), 'a') as file:
        file.write("---------------------\n")
        file.write("Complete:     "+ format((benchstop - benchstart), '.3f') + 's')
