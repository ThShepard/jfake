# jfake
Python CLI tool to visually detect photoshopped pictures using ELA  

## usage:  
jfake.py [-h] --input INPUT [--output OUTPUT] [--verbose] [--debug] [--quality QUALITY] [--multiplier MULTIPLIER]  [--benchmark] [--entropy] [--psnr] [--numba]

## Required arguments:  
  --input STR, -i STR       Input image file [BMP], [GIF], [JPEG], [PNG], [PPM], [TIFF]

## Optional arguments:  
  -h, --help                show this help message and exit  
  --output PATH, -o PATH    Define output folder jfake (default: output)  
  --verbose, -v             Write all steps to terminal (default: False)  
  --debug, -d               Write all steps to output folder (default: False)  
  --quality INT, -q INT     JPEG-Quality [1-99] jfake (default: 50)  
  --multiplier INT, -m INT  Multiplier jfake (default: Automatic)  
  --benchmark, -b           Write needed time per step in file (default: False)  
  --entropy, -e             Calculate entropy in each processing step (default: False)  
  --psnr, -p                Calculate signal-to-noise-ratio (PSNR) (default: False)  
  --numba, -n               Use numba jit compiler for better performance (default: False)  

## Examples:  
jfake.py -i lenna.png  
jfake.py --input lenna.png --entropy  
jfake.py -i lenna.png -o lenna -vdbepn  