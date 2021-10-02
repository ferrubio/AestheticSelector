import wx
import os
import pickle
import glob
import numpy as np
from shutil import copyfile, move

from threading import Thread
from wx.lib.pubsub import pub

import tensorflow as tf
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import imread
from matplotlib import cm

def read_image(image_path, label):
    """
      Cargamos una imagen usando su ruta (path), la convertimos en tensor y la normalizamos
    """
    contents = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(contents, channels=3)
    img = tf.cast(img, tf.float64)
    img /= 255.0
    return img, label

def resize_image(img, label, target_size):
    """
      Redimensionamos una imagen
    """
    resized_img = tf.image.resize(img, target_size)
    return resized_img, label

def get_dataset(image_paths, image_labels, target_size, batch_size, prep_func=None):
    """
      - Generamos un objeto tf.data.Dataset para optimizar el entrenamiento desde los
        paths de las imagenes
      - Aplicamos las funciones read_image y resize_image a las imagenes
      - Podemos usar una función prep_func si queremos hacer fine-tunning
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    dataset = dataset.map(read_image)
    dataset = dataset.map(lambda x, y: resize_image(x, y, target_size))

    if prep_func != None:
        dataset = dataset.map(lambda x, y: (x*255.0, y))
        dataset = dataset.map(lambda x, y: (prep_func(x), y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def transform_img_fn(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(299, 299))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x


########################################################################
class ProgressBar(tf.keras.callbacks.Callback):

    def on_batch_end(self, batch, logs={}):
        wx.CallAfter(pub.sendMessage, "update", msg="")

class TestThread(Thread):
    """Test Worker Thread Class."""

    # ----------------------------------------------------------------------
    def __init__(self, iterator, files, outputDirectory, slider, copy=True):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.mainIterator = iterator
        self.files = files
        self.outputDirectory = outputDirectory
        self.sliderValue = slider
        self.copy = copy

        tf.keras.backend.clear_session()
        self.start()

    # ----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        with open("Model_definition.pkl", 'rb') as f:
            model_config = pickle.load(f)

        model = tf.keras.models.Model.from_config(model_config)
        model.load_weights("Model_weights.h5")

        wx.CallAfter(pub.sendMessage, "update", msg="Making Predictions")
        # El callback aun no funciona, pero para cuando este disponible. De momento hay 3 saltos, uno que carga el
        # modelo, otro procesa las imagenes y el ultimo ordena y copia.
        my_call = ProgressBar()
        predictions = model.predict(self.mainIterator)
        wx.CallAfter(pub.sendMessage, "update", msg="Selecting best images")
        predictions = predictions[:, 1]
        order_pred = np.sort(predictions)[::-1]
        order_args = np.argsort(predictions)[::-1]
        subset_images = np.array(self.files)[order_args[0:len(self.files) * self.sliderValue // 100]]
        for idx, x in enumerate(subset_images):
            if self.copy:
                copyfile(x, os.path.join(self.outputDirectory, "{}_{:.3f}_{}".format(idx, order_pred[idx], x.split('/')[-1])))
            else:
                move(x, os.path.join(self.outputDirectory, "{}_{:.3f}_{}".format(idx, order_pred[idx], x.split('/')[-1])))
        wx.CallAfter(pub.sendMessage, "update", msg="")

class MyProgressDialog(wx.Dialog):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, parent, title):
        """Constructor"""
        wx.Dialog.__init__(self, parent, title = title)
        panel = wx.Panel(self)
        self.count = 0

        self.progress = wx.Gauge(panel, range=2)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add((-1, 50))
        sizer.Add(self.progress, 0, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=20)

        self.st2 = wx.StaticText(panel, label='Loading model')
        sizer.Add(self.st2, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=20)

        panel.SetSizer(sizer)

        # create a pubsub receiver
        pub.subscribe(self.updateProgress, "update")

    # ----------------------------------------------------------------------
    def updateProgress(self, msg):
        """"""
        self.count += 1

        if self.count > 2:
            self.EndModal(0)

        self.progress.SetValue(self.count)
        self.st2.SetLabel(msg)

########################################################################
class ProgressBar2(tf.keras.callbacks.Callback):

    def on_batch_end(self, batch, logs={}):
        wx.CallAfter(pub.sendMessage, "update", msg="")

class TestThread2(Thread):
    """Test Worker Thread Class."""

    # ----------------------------------------------------------------------
    def __init__(self, file, info_type):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.file = file
        self.info_type = info_type
        self.saliency_maps = 0
        self.gradcam_maps = 0

        tf.keras.backend.clear_session()
        self.start()

    # ----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        with open("Model_definition.pkl", 'rb') as f:
            model_config = pickle.load(f)

        model = tf.keras.models.Model.from_config(model_config)
        model.load_weights("Model_weights.h5")

        wx.CallAfter(pub.sendMessage, "update", msg="Preparing Image")
        # El callback aun no funciona, pero para cuando este disponible. De momento hay 3 saltos, uno que carga el
        # modelo, otro procesa las imagenes y el ultimo ordena y copia.
        my_call = ProgressBar2()
        img_prep = transform_img_fn(self.file)
        img_prep = np.concatenate([img_prep,img_prep])

        if self.info_type == 'SmoothGrad Saliency':
            wx.CallAfter(pub.sendMessage, "update", msg="Obtaining SmoothGrad Saliency")
            replace2linear = ReplaceToLinear()
            score = CategoricalScore([0, 1])
            saliency = Saliency(model, model_modifier=replace2linear, clone=True)
            self.saliency_maps = saliency(score, img_prep, smooth_samples=50, smooth_noise=0.10)

        else:
            wx.CallAfter(pub.sendMessage, "update", msg="Obtaining GradCAM++")
            replace2linear = ReplaceToLinear()
            score = CategoricalScore([0,1])
            gradcam = GradcamPlusPlus(model, model_modifier=replace2linear,clone=True)
            self.gradcam_maps = gradcam(score, img_prep, penultimate_layer=-1)

        wx.CallAfter(pub.sendMessage, "update", msg="")


########################################################################
class MyProgressDialog2(wx.Dialog):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, parent, title):
        """Constructor"""
        wx.Dialog.__init__(self, parent, title = title)
        panel = wx.Panel(self)
        self.count = 0

        self.progress = wx.Gauge(panel, range=2)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add((-1, 50))
        sizer.Add(self.progress, 0, flag=wx.LEFT | wx.RIGHT | wx.EXPAND, border=20)

        self.st2 = wx.StaticText(panel, label='Loading model')
        sizer.Add(self.st2, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=20)

        panel.SetSizer(sizer)

        # create a pubsub receiver
        pub.subscribe(self.updateProgress, "update")

    # ----------------------------------------------------------------------
    def updateProgress(self, msg):
        """"""
        self.count += 1

        if self.count > 2:
            self.EndModal(0)

        self.progress.SetValue(self.count)
        self.st2.SetLabel(msg)

class TabFilter(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.currentDirectory = os.getcwd()
        self.inputDirectory = ""
        self.outputDirectory = ""
        self.num_images = 0

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(10)

        # we create a main horizontal box to attach the intruction image in the right.
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        vbox = wx.BoxSizer(wx.VERTICAL)
        ################## INPUT FOLDER ###########################
        # titulo
        st1 = wx.StaticText(self, label='Select Input Folder')
        vbox.Add(st1, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=20)
        # button
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        input_button = wx.Button(self, label="Choose Folder")
        input_button.Bind(wx.EVT_BUTTON, self.onDirIn)
        hbox1.Add(input_button, 0, wx.ALL | wx.CENTER, 5)
        # texto para la carpeta
        self.inputText = wx.StaticText(self, label='')
        self.inputText.SetFont(font)
        hbox1.Add(self.inputText, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=8)
        vbox.Add(hbox1, flag=wx.EXPAND | wx.LEFT, border=20)
        # texto para el generador
        self.genText = wx.StaticText(self, label='')
        vbox.Add(self.genText, flag=wx.EXPAND | wx.LEFT, border=25)


        ################## OUTPUT FOLDER ###########################
        # titulo
        st2 = wx.StaticText(self, label='Select Output Folder')
        vbox.Add(st2, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=20)
        # button
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        output_button = wx.Button(self, label="Choose Folder")
        output_button.Bind(wx.EVT_BUTTON, self.onDirOut)
        hbox2.Add(output_button, 0, wx.ALL | wx.CENTER, 5)
        # texto para la carpeta
        self.outputText = wx.StaticText(self, label='')
        self.outputText.SetFont(font)
        hbox2.Add(self.outputText, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=8)
        vbox.Add(hbox2, flag=wx.EXPAND | wx.LEFT, border=20)
        # texto para el generador
        self.outGenText = wx.StaticText(self, label='')
        vbox.Add(self.outGenText, flag=wx.EXPAND | wx.LEFT, border=25)
        vbox.Add((-1, 25))

        ################## CHECKBOX AND PERCENTAGE ##################
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        # Slider
        st3 = wx.StaticText(self, label='Percentage of images')
        hbox3.Add(st3, flag=wx.LEFT, border=0)
        sl3 = wx.Slider(self, value=10, minValue=0, maxValue=100)
        sl3.Bind(wx.EVT_SLIDER, self.onSlider)
        hbox3.Add(sl3, flag=wx.LEFT, border=10)
        self.sliderValue = sl3.GetValue()
        self.sliderText = wx.StaticText(self, label=str(self.sliderValue)+" %")
        hbox3.Add(self.sliderText, flag=wx.LEFT, border=10)
        vbox.Add(hbox3, flag=wx.LEFT, border=20)
        vbox.Add((-1, 25))
        # Checkbox
        cb3 = wx.CheckBox(self, label='Copy images')
        self.checkboxValue = True
        cb3.SetValue(self.checkboxValue)
        cb3.Bind(wx.EVT_CHECKBOX, self.onCheckbox)
        vbox.Add(cb3, flag=wx.LEFT, border=20)
        vbox.Add((-1, 50))

        ################## BUTTONS ##################
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        #btn2 = wx.Button(self, label='Close')
        #btn2.Bind(wx.EVT_BUTTON, self.onClose)
        #hbox4.Add(btn2, flag=wx.EXPAND)
        btn1 = wx.Button(self, label='Process', size=(150, 60))
        btn1.Bind(wx.EVT_BUTTON, self.onProcess)
        hbox4.Add(btn1, flag=wx.EXPAND | wx.LEFT, border=10)
        vbox.Add(hbox4, flag=wx.LEFT, border=100)

        hbox.Add(vbox, flag=wx.LEFT, border=5)
        hbox.Add((40, -1))

        figure = Figure(figsize=(5.5, 4))
        axes = figure.add_subplot(111)
        axes.axis('off')
        axes.imshow(imread('instructions.png'))
        figure.tight_layout(pad=0)
        canvas = FigureCanvas(self, -1, figure)
        #img = wx.Image('instructions.png', wx.BITMAP_TYPE_ANY)
        # scale the image, preserving the aspect ratio
        '''
        W = img.GetWidth()
        H = img.GetHeight()
        main_size = 408
        if W > H:
            NewW = main_size
            NewH = main_size * H / W
        else:
            NewH = main_size
            NewW = main_size * W / H
        img = img.Scale(NewW, NewH)
        '''

        #imageIns = wx.StaticBitmap(self, wx.ID_ANY,
        #                                 wx.Bitmap(img))
        hbox.Add(canvas, 0, wx.ALL, 5)

        self.SetSizer(hbox)

    # ----------------------------------------------------------------------
    def onDirIn(self, event):
        """
        Show the DirDialog and print the user's choice to stdout
        """
        dlg = wx.DirDialog(self, "Choose a directory:",
                           style=wx.DD_DEFAULT_STYLE
                           # | wx.DD_DIR_MUST_EXIST
                           # | wx.DD_CHANGE_DIR
                           )
        if dlg.ShowModal() == wx.ID_OK:
            self.inputDirectory = dlg.GetPath()
            self.inputText.SetLabel(self.inputDirectory)
            self.inputText.SetForegroundColour("black") # set text color

            mylist = [f for f in glob.glob(os.path.join(self.inputDirectory, "*.jpg"))]
            print(not mylist)
            if not mylist:
                self.genText.SetLabel("* Valid images not found")
                self.genText.SetForegroundColour("red")
                self.inputDirectory = ""

            else:
                self.files = mylist
                myNames = [os.path.split(f)[1] for f in mylist]

                self.num_images = len(mylist)
                self.mainIterator = get_dataset(mylist, np.zeros(self.num_images), (299,299), 1,
                                                tf.keras.applications.inception_v3.preprocess_input)

                self.genText.SetLabel("* Found {} validated images".format(self.num_images))
                self.genText.SetForegroundColour("black")

                if not self.num_images * self.sliderValue // 100:
                    self.outGenText.SetLabel("* No images will be saved. Percentage too low.")
                    self.outGenText.SetForegroundColour("red")
                else:
                    self.outGenText.SetLabel(
                        "* Saving {} best images".format(self.num_images * self.sliderValue // 100))
                    self.outGenText.SetForegroundColour("black")
        dlg.Destroy()

    # ----------------------------------------------------------------------
    def onDirOut(self, event):
        """
        Show the DirDialog and print the user's choice to stdout
        """
        dlg = wx.DirDialog(self, "Choose a directory:",
                           style=wx.DD_DEFAULT_STYLE
                           # | wx.DD_DIR_MUST_EXIST
                           # | wx.DD_CHANGE_DIR
                           )
        if dlg.ShowModal() == wx.ID_OK:
            self.outputDirectory = dlg.GetPath()
            self.outputText.SetLabel(self.outputDirectory)
            self.outputText.SetForegroundColour("black")  # set text color
        dlg.Destroy()

    # ----------------------------------------------------------------------
    def onSlider(self,event):
        obj = event.GetEventObject()
        self.sliderValue = obj.GetValue()
        self.sliderText.SetLabel(str(self.sliderValue)+" %")

        if not self.num_images * self.sliderValue // 100:
            self.outGenText.SetLabel("* No images will be saved. Percentage too low.".format(self.num_images))
            self.outGenText.SetForegroundColour("red")
        else:
            self.outGenText.SetLabel("* Saving {} best images".format(self.num_images * self.sliderValue // 100))
            self.outGenText.SetForegroundColour("black")

    # ----------------------------------------------------------------------
    def onCheckbox(self,event):
        obj = event.GetEventObject()
        self.checkboxValue = obj.GetValue()

    # ----------------------------------------------------------------------

    def onProcess(self, event):
        """
        Runs the thread
        """
        pass_checks = True
        # Comprobamos rutas y si hay imagenes
        if not self.inputDirectory:
            self.inputText.SetLabel("* This field is required.")
            self.inputText.SetForegroundColour("red")  # set text color
            pass_checks = False
        else:
            if not self.num_images:
                pass_checks = False

            if not self.num_images * self.sliderValue // 100:
                pass_checks = False

        if not self.outputDirectory:
            self.outputText.SetLabel("* This field is required.")
            self.outputText.SetForegroundColour("red")  # set text color
            pass_checks = False

        if not pass_checks:
            return

        TestThread(self.mainIterator, self.files, self.outputDirectory, self.sliderValue, self.checkboxValue)
        dlg = MyProgressDialog(self, "Progress")
        dlg.ShowModal()

        self.Clean()

    def Clean(self):
        self.inputText.SetLabel("")
        self.inputText.SetForegroundColour("black")  # set text color
        self.outputText.SetLabel("")
        self.outputText.SetForegroundColour("black")  # set text color
        self.genText.SetLabel("")
        self.genText.SetForegroundColour("black")
        self.outGenText.SetLabel("")
        self.outGenText.SetForegroundColour("black")
        self.outputDirectory = ""
        self.inputDirectory = ""

class TabInterpretability(wx.Panel):

    def __init__(self, parent):
        self.panel = wx.Panel.__init__(self, parent)

        self.PhotoMaxSize = 299

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(10)

        # main vbox
        vbox = wx.BoxSizer(wx.VERTICAL)
        # horizontal box for images
        hbox0 = wx.BoxSizer(wx.HORIZONTAL)

        #vbox.Add(st1, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=20)
        # imagen
        vbox1 = wx.BoxSizer(wx.VERTICAL)
        st1 = wx.StaticText(self, label='Resized Image')
        vbox1.Add(st1)
        figure = Figure(figsize=(3, 3))
        self.axes = figure.add_subplot(111)
        self.axes.axis('off')
        figure.tight_layout(pad=0)
        self.canvas = FigureCanvas(self, -1, figure)
        vbox1.Add(self.canvas, flag=wx.TOP, border=10)
        hbox0.Add(vbox1, 0, wx.ALL, 5)

        # bad class
        vbox2 = wx.BoxSizer(wx.VERTICAL)
        st2 = wx.StaticText(self, label='Bad Quality Information')
        vbox2.Add(st2)
        bad_figure = Figure(figsize=(3, 3))
        self.bad_axes = bad_figure.add_subplot(111)
        self.bad_axes.axis('off')
        bad_figure.tight_layout(pad=0)
        self.bad_canvas = FigureCanvas(self, -1, bad_figure)
        vbox2.Add(self.bad_canvas, flag=wx.TOP, border=10)
        hbox0.Add(vbox2, 0, wx.ALL, 5)

        # good class
        vbox3 = wx.BoxSizer(wx.VERTICAL)
        st3 = wx.StaticText(self, label='Good Quality Information')
        vbox3.Add(st3)
        good_figure = Figure(figsize=(3, 3))
        self.good_axes = good_figure.add_subplot(111)
        self.good_axes.axis('off')
        good_figure.tight_layout(pad=0)
        self.good_canvas = FigureCanvas(self, -1, good_figure)
        vbox3.Add(self.good_canvas, flag=wx.TOP, border=10)
        hbox0.Add(vbox3, 0, wx.ALL, 5)

        vbox.Add(hbox0, flag=wx.EXPAND | wx.LEFT, border=20)

        vbox.Add((-1, 25))
        # button
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        input_button = wx.Button(self, label="Browse Image")
        input_button.Bind(wx.EVT_BUTTON, self.onBrowse)
        hbox1.Add(input_button, 0, wx.ALL | wx.CENTER, 5)
        # texto para la carpeta
        self.inputText = wx.StaticText(self, label='')
        self.inputText.SetFont(font)
        hbox1.Add(self.inputText, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=8)

        st4 = wx.StaticText(self, label='Select Image Information:')
        hbox1.Add(st4,flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=450)
        self.dropdown = wx.ComboBox(self, value="SmoothGrad Saliency", choices=['SmoothGrad Saliency', 'GradCAM++'], style=wx.CB_READONLY)
        hbox1.Add(self.dropdown, flag=wx.LEFT, border=10)

        vbox.Add(hbox1, flag=wx.EXPAND | wx.LEFT, border=20)

        self.SetSizer(vbox)


    def onBrowse(self, event):
        """
        Browse for file
        Show the DirDialog and print the user's choice to stdout
        """
        dlg = wx.FileDialog(self, "Choose an image:",
                           style=wx.DD_DEFAULT_STYLE
                           # | wx.DD_DIR_MUST_EXIST
                           # | wx.DD_CHANGE_DIR
                           )
        if dlg.ShowModal() == wx.ID_OK:
            self.inputImg = dlg.GetPath()
            self.inputText.SetLabel(self.inputImg)
            self.inputText.SetForegroundColour("black")  # set text color
            self.onView()
        dlg.Destroy()

    def onView(self):
        filepath = self.inputImg
        self.axes.imshow(imread(self.inputImg), aspect='auto')
        self.canvas.draw()

        tt2 = TestThread2(filepath, self.dropdown.GetValue())
        dlg = MyProgressDialog2(self, "Progress")
        dlg.ShowModal()

        if self.dropdown.GetValue() == 'SmoothGrad Saliency':
            self.bad_axes.imshow(tt2.saliency_maps[0], cmap='jet')
            self.bad_canvas.draw()

            self.good_axes.imshow(tt2.saliency_maps[1], cmap='jet')
            self.good_canvas.draw()
        else:
            bad_heatmap = np.uint8(cm.jet(tt2.gradcam_maps[0])[..., :3] * 255)
            # self.bad_axes.imshow(imread(filepath), aspect='auto')
            self.bad_axes.imshow(bad_heatmap, cmap='jet')
            self.bad_canvas.draw()

            good_heatmap = np.uint8(cm.jet(tt2.gradcam_maps[1])[..., :3] * 255)
            # self.good_axes.imshow(imread(filepath), aspect='auto')
            self.good_axes.imshow(good_heatmap, cmap='jet')
            self.good_canvas.draw()


class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title='Aesthetic Selector', size=(1000, 500))

        # Create a panel and notebook (tabs holder)
        p = wx.Panel(self)
        nb = wx.Notebook(p)

        # Create the tab windows
        tab1 = TabFilter(nb)
        tab2 = TabInterpretability(nb)

        # Add the windows to tabs and name them.
        nb.AddPage(tab1, "        Filter        ")
        nb.AddPage(tab2, "Interpretability")

        # Set noteboook in a sizer to create the layout
        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        p.SetSizer(sizer)

        #self.Layout()
        self.Show()

########################################################################
if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()