# AestheticSelector

## Instalation

Two packages where created:

[Mac OS > 10.13 - High Sierra](https://www.dropbox.com/s/2mvmdyhgovz0fiv/AestheticSelector.zip?dl=0)

[Windows 10 version](https://www.dropbox.com/s/14tbyzi910p8cdl/AestheticSelector.exe)

If you have a different OS or the packages are not working, you can download the repository and launch the application with python3.6 or higher:
* Clone the repository.
* Create a virtual env and install the requirements.
* Download the [weights](https://www.dropbox.com/s/vnpmjttvylqq9dx/Model_weights.h5) in the clone repository.
* Run AestheticSelector.py

## Usage

* Select the folder with the images (actual version only support .jpg format). Then, valid images are identified.
* Select the folder where save the best images based on their aesthetic quality. You can choose between copying or moving the images.
* Select the percentage to store, from the valid images detected.
* Launch the model with the "Process" button.
