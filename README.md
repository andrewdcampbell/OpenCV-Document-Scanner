# Document Scanner

### An interactive document scanner built in Python using OpenCV

The scanner takes a poorly scanned image, finds the corners of the document, applies the perspective transformation to get a top-down view of the document, sharpens the image, and applies an adaptive color threshold to clean up the image.

On my test dataset of 280 images, the program correctly detected the corners of the document 92.8% of the time.

This project makes use of the transform and imutils modules from pyimagesearch (which can be accessed [here](http://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/)). The UI code for the interactive mode is adapted from `poly_editor.py` from [here](https://matplotlib.org/examples/event_handling/poly_editor.html).

* You can manually click and drag the corners of the document to be perspective transformed:
![Example of interactive GUI](https://github.com/andrewdcampbell/doc_scanner/blob/master/ui.gif)

* The scanner can also process an entire directory of images automatically and save the output in an output directory:
![Image Directory of images to be processed](https://github.com/andrewdcampbell/doc_scanner/blob/master/before_after.gif)

#### Here are some examples of images before and after scan:
<img src="https://github.com/andrewdcampbell/doc_scanner/blob/master/sample_images/cell_pic.jpg" height="450"> <img src="https://github.com/andrewdcampbell/doc_scanner/blob/master/output/cell_pic.jpg" height="450">

<img src="https://github.com/andrewdcampbell/doc_scanner/blob/master/sample_images/receipt.jpg" height="450"> <img src="https://github.com/andrewdcampbell/doc_scanner/blob/master/output/receipt.jpg" height="450">

<img src="https://github.com/andrewdcampbell/doc_scanner/blob/master/sample_images/math_cheat_sheet.JPG" height="450"> <img src="https://github.com/andrewdcampbell/doc_scanner/blob/master/output/math_cheat_sheet.JPG" height="450">

<img src="https://github.com/andrewdcampbell/doc_scanner/blob/master/sample_images/dollar_bill.JPG" width="350"> <img src="https://github.com/andrewdcampbell/doc_scanner/blob/master/output/dollar_bill.JPG" width="350">


### Usage
```
python scan.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
```
* The `-i` flag enables interactive mode, where you will be prompted to click and drag the corners of the document. For example, to scan a single image with interactive mode enabled:
```
python scan.py --image sample_images/desk.JPG -i
```
* Alternatively, to scan all images in a directory without any input:
```
python scan.py --images sample_images
```
