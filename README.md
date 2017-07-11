# Document Scanner

### An interactive document scanner built in Python using OpenCV

The scanner takes a poorly scanned image, finds the corners of the document, applies the perspective transformation to get a top-down view of the document, sharpens the image, and applies an adaptive color threshold to clean up the image.

On my test dataset of 280 images, the program correctly detected the corners of the document 92.8% of the time.

This project makes use of the transform and imutils modules from pyimagesearch (which can be accessed [here](http://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/)). The UI code for the interactive mode is adapted from `poly_editor.py` from [here](https://matplotlib.org/examples/event_handling/poly_editor.html).

* You can manually click and drag the corners of the document to be perspective transformed:
![Example of interactive GUI](https://github.com/andrewdcampbell/doc_scanner/blob/master/ui.gif)

* The scanner can also process an entire directory of images automatically and save the output in an output directory:
![Image Directory of images to be processed](https://github.com/andrewdcampbell/doc_scanner/blob/master/before_after.gif)


### Usage
```
python scan.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
```
* For example, to scan a single image with interactive mode:
```
python scan.py --image images/page.jpg -i
```
* To scan all images in a directory automatically:
```
python scan.py --images images
```