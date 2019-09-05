import wx
import os
import pickle
import glob
import pandas as pd
import numpy as np
from shutil import copyfile, move

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras
import keras_applications
from keras import backend as K

keras_applications.set_keras_submodules(backend=keras.backend,
                                        layers=keras.layers,
                                        models=keras.models,
                                        utils=keras.utils)

from threading import Thread
from wx.lib.pubsub import pub

class ProgressBar(keras.callbacks.Callback):

    def on_batch_end(self, batch, logs={}):
        wx.CallAfter(pub.sendMessage, "update", msg="")

########################################################################
class TestThread(Thread):
    """Test Worker Thread Class."""

    # ----------------------------------------------------------------------
    def __init__(self, iterator, frame, outputDirectory, slider, copy=True):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.mainIterator = iterator
        self.frame = frame
        self.outputDirectory = outputDirectory
        self.sliderValue = slider
        self.copy = copy

        K.clear_session()
        self.start()

    # ----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        with open("Keras_FinetuningAVA_inception_binaryWeights_mse_Adam_LR1e-05_BS64_E20_model.pkl", 'rb') as f:
            model_config = pickle.load(f)

        model = Model.from_config(model_config)
        model.load_weights("Keras_FinetuningAVA_inception_binaryWeights_mse_Adam_LR1e-05_BS64_E20_bestmodel0.h5")

        wx.CallAfter(pub.sendMessage, "update", msg="Making Predictions")
        # El callback aun no funciona, pero para cuando este disponible. De momento hay 3 saltos, uno que carga el
        # modelo, otro procesa las imagenes y el ultimo ordena y copia.
        my_call = ProgressBar()
        predictions = model.predict_generator(self.mainIterator,
                                              steps=len(self.mainIterator)
                                              #, callbacks = [my_call]
                                              )
        wx.CallAfter(pub.sendMessage, "update", msg="Selecting best images")
        predictions = predictions[:,1]
        order_pred = np.sort(predictions)[::-1]
        order_args = np.argsort(predictions)[::-1]
        subset_images = np.array(self.frame.loc[order_args[0:self.mainIterator.n * self.sliderValue // 100]])
        for idx, x in enumerate(subset_images):
            if self.copy:
                copyfile(x[0], os.path.join(self.outputDirectory, "{}_{}_{}".format(idx, order_pred[idx], x[1])))
            else:
                move(x[0], os.path.join(self.outputDirectory, "{}_{}_{}".format(idx, order_pred[idx], x[1])))
        wx.CallAfter(pub.sendMessage, "update", msg="")


########################################################################
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



class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Aesthetic Filter', size=(590, 360))
        panel = wx.Panel(self)

        self.currentDirectory = os.getcwd()
        self.inputDirectory = ""
        self.outputDirectory = ""

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        font.SetPointSize(10)

        vbox = wx.BoxSizer(wx.VERTICAL)


        ################## INPUT FOLDER ###########################
        # titulo
        st1 = wx.StaticText(panel, label='Select Input Folder')
        vbox.Add(st1, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=20)
        # button
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        input_button = wx.Button(panel, label="Choose Folder")
        input_button.Bind(wx.EVT_BUTTON, self.onDirIn)
        hbox1.Add(input_button, 0, wx.ALL | wx.CENTER, 5)
        # texto para la carpeta
        self.inputText = wx.StaticText(panel, label='')
        self.inputText.SetFont(font)
        hbox1.Add(self.inputText, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=8)
        vbox.Add(hbox1, flag=wx.EXPAND | wx.LEFT, border=20)
        # texto para el generador
        self.genText = wx.StaticText(panel, label='')
        vbox.Add(self.genText, flag=wx.EXPAND | wx.LEFT, border=25)


        ################## OUTPUT FOLDER ###########################
        # titulo
        st2 = wx.StaticText(panel, label='Select Output Folder')
        vbox.Add(st2, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=20)
        # button
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        output_button = wx.Button(panel, label="Choose Folder")
        output_button.Bind(wx.EVT_BUTTON, self.onDirOut)
        hbox2.Add(output_button, 0, wx.ALL | wx.CENTER, 5)
        # texto para la carpeta
        self.outputText = wx.StaticText(panel, label='')
        self.outputText.SetFont(font)
        hbox2.Add(self.outputText, flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=8)
        vbox.Add(hbox2, flag=wx.EXPAND | wx.LEFT, border=20)
        # texto para el generador
        self.outGenText = wx.StaticText(panel, label='')
        vbox.Add(self.outGenText, flag=wx.EXPAND | wx.LEFT, border=25)

        vbox.Add((-1, 25))
        ################## CHECKBOX AND PERCENTAGE ##################
        # Checkbox
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        cb3 = wx.CheckBox(panel, label='Copy images')
        self.checkboxValue = True
        cb3.SetValue(self.checkboxValue)
        cb3.Bind(wx.EVT_CHECKBOX, self.onCheckbox)
        hbox3.Add(cb3)
        # Slider
        st3 = wx.StaticText(panel, label='Percentage of images')
        hbox3.Add(st3, flag=wx.LEFT, border=50)
        sl3 = wx.Slider(panel, value=10, minValue=0, maxValue=100)
        sl3.Bind(wx.EVT_SLIDER, self.onSlider)
        hbox3.Add(sl3, flag=wx.LEFT, border=10)
        self.sliderValue = sl3.GetValue()
        self.sliderText = wx.StaticText(panel, label=str(self.sliderValue)+" %")
        hbox3.Add(self.sliderText, flag=wx.LEFT, border=10)
        vbox.Add(hbox3, flag=wx.LEFT, border=20)

        vbox.Add((-1, 50))
        ################## BUTTONS ##################
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        btn2 = wx.Button(panel, label='Close')
        btn2.Bind(wx.EVT_BUTTON, self.onClose)
        hbox4.Add(btn2, flag=wx.EXPAND)
        btn1 = wx.Button(panel, label='Process')
        btn1.Bind(wx.EVT_BUTTON, self.onProcess)
        hbox4.Add(btn1, flag=wx.EXPAND | wx.LEFT, border=35)
        vbox.Add(hbox4, flag=wx.LEFT, border=310)

        panel.SetSizer(vbox)
        self.Show()

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
                myNames = [os.path.split(f)[1] for f in mylist]

                self.frame = pd.DataFrame(mylist, columns=['files'])
                self.frame['names'] = myNames
                datagen = ImageDataGenerator(preprocessing_function=keras_applications.inception_v3.preprocess_input)
                self.mainIterator = datagen.flow_from_dataframe(self.frame,
                                                                x_col='files',
                                                                target_size=(299, 299),
                                                                class_mode=None,
                                                                batch_size=1,
                                                                shuffle=False)
                if not self.mainIterator.n:
                    self.genText.SetLabel("* Valid images not found")
                    self.genText.SetForegroundColour("red")
                else:
                    self.genText.SetLabel("* Found {} validated images".format(self.mainIterator.n))
                    self.genText.SetForegroundColour("black")

                if not self.mainIterator.n * self.sliderValue // 100:
                    self.outGenText.SetLabel("* No images will be saved. Percentage too low.")
                    self.outGenText.SetForegroundColour("red")
                else:
                    self.outGenText.SetLabel(
                        "* Saving {} best images".format(self.mainIterator.n * self.sliderValue // 100))
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

        if not self.mainIterator.n * self.sliderValue // 100:
            self.outGenText.SetLabel("* No images will be saved. Percentage too low.".format(self.mainIterator.n))
            self.outGenText.SetForegroundColour("red")
        else:
            self.outGenText.SetLabel("* Saving {} best images".format(self.mainIterator.n * self.sliderValue // 100))
            self.outGenText.SetForegroundColour("black")

    # ----------------------------------------------------------------------
    def onCheckbox(self,event):
        obj = event.GetEventObject()
        self.checkboxValue = obj.GetValue()

    # ----------------------------------------------------------------------
    def onClose(self, event):
        """"""
        self.Close()

    # ----------------------------------------------------------------------
    '''
    def onProcess(self, event):
        """"""
        with open("Keras_FinetuningAVA_inception_binaryWeights_mse_Adam_LR1e-05_BS64_E20_model.pkl", 'rb') as f:
            model_config = pickle.load(f)
        model = Model.from_config(model_config)
        model.load_weights("Keras_FinetuningAVA_inception_binaryWeights_mse_Adam_LR1e-05_BS64_E20_bestmodel0.h5")
        predictions = model.predict_generator(self.mainIterator,
                                              steps=len(self.mainIterator))
        predictions = predictions[:,1]
        order_args = np.argsort(predictions)[::-1]

        subset_images = np.array(self.frame.loc[order_args[0:self.mainIterator.n * self.sliderValue // 100]])

        for idx, x in enumerate(subset_images):
            copyfile(x[0], os.path.join(self.outputDirectory, "{}_{}".format(idx, x[1])))
    '''
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
            if not self.mainIterator.n:
                pass_checks = False

            if not self.mainIterator.n * self.sliderValue // 100:
                pass_checks = False

        if not self.outputDirectory:
            self.outputText.SetLabel("* This field is required.")
            self.outputText.SetForegroundColour("red")  # set text color
            pass_checks = False

        if not pass_checks:
            return

        TestThread(self.mainIterator, self.frame, self.outputDirectory, self.sliderValue, self.checkboxValue)
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


########################################################################
if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()