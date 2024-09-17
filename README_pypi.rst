=====================================
Libreface
=====================================

`GitHub Repo`_ - https://github.com/ihp-lab/LibreFace

.. _`GitHub Repo`: https://github.com/ihp-lab/LibreFace

|badge1| |badge2|


.. |badge1| image:: https://img.shields.io/badge/version-0.0.16-blue
   :alt: Static Badge


.. |badge2| image:: https://img.shields.io/badge/python-%3E%3D3.8-green
   :alt: Static Badge


This is the python package for `LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis`_.
LibreFace is an open-source and comprehensive toolkit for accurate and real-time facial expression analysis with both CPU and GPU acceleration versions.
LibreFace eliminates the gap between cutting-edge research and an easy and free-to-use non-commercial toolbox. We propose to adaptively pre-train the vision encoders with various face datasets and then distillate them to a lightweight ResNet-18 model in a feature-wise matching manner.
We conduct extensive experiments of pre-training and distillation to demonstrate that our proposed pipeline achieves comparable results to state-of-the-art works while maintaining real-time efficiency.

.. _`LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis`: https://boese0601.github.io/libreface



Dependencies
============

- Python>=3.8
- You should have `cmake` installed in your system.
    - **For Linux users** - :code:`sudo apt-get install cmake`. If you run into troubles, consider upgrading to the latest version (`instructions`_).
    - **For Mac users** - :code:`brew install cmake`.

.. _`instructions`: https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line


Installation
============
You can install this package using `pip` from the testPyPI hub:

.. code-block:: bash

    python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple libreface==0.0.16


Usage
=====

Commandline
----------------

You can use this package through commandline using the following command.

.. code-block:: bash

    libreface --input_path="path/to/your_image_or_video.jpg"

Note that the above script would save results in a CSV at the default location - :code:`sample_results.csv`. If you want to specify your own path, use the :code:`--output_path`  commandline argument,

.. code-block:: bash

    libreface --input_path="path/to/your_image_or_video.jpg" --output_path="path/to/save_results.csv"

To change the device used for inference, use the :code:`--device` commandline argument,

.. code-block:: bash

    libreface --input_path="path/to/your_image_or_video.jpg" --device="cuda:0"

To save intermediate files, :code:`libreface` uses a temporary directory that defaults to ./tmp. To change the temporary directory path,

.. code-block:: bash

    libreface --input_path="path/to/your_image_or_video.jpg" --temp="your/temp/path"

**Examples**

Download a `sample image`_ from our github repository. To get the facial attributes for this image and save to a CSV file simply run,

.. _`sample image`: https://github.com/ihp-lab/LibreFace/blob/pypi_wrap/sample_disfa.png

.. code-block:: bash

    libreface --input_path="sample_disfa.png"

Download a `sample video`_ from our github repository. To run the inference on this video using a GPU and save the results to :code:`my_custom_file.csv` run the following command,

.. _`sample video`: https://github.com/ihp-lab/LibreFace/blob/pypi_wrap/sample_disfa.avi

.. code-block:: bash
    
    libreface --input_path="sample_disfa.avi" --output_path="my_custom_file.csv" --device="cuda:0"

Note that for videos, each row in the saved csv file correspond to individual frames in the given video.

Python API
--------------

Here’s how to use this package in your python scripts.

To assign the results to a python variable,

.. code-block:: python

    import libreface 
    detected_attributes = libreface.get_facial_attributes(image_or_video_path)

To save the results to a csv file, 

.. code-block:: python

    import libreface 
    libreface.get_facial_attributes(image_or_video_path,
                                    output_save_path = "your_save_path.csv")

To change the device used for inference, use the :code:`device` parameter,

.. code-block:: python

    import libreface 
    libreface.get_facial_attributes(image_or_video_path,
                                    device = "cuda:0") # can be "cpu" or "cuda:0", "cuda:1", ...

To save intermediate files, libreface uses a temporary directory that defaults to :code:`./tmp`. To change the temporary directory path,

.. code-block:: python

    import libreface 
    libreface.get_facial_attributes(image_or_video_path,
                                    temp_dir = "your/temp/path") 

Downloading Model Weights
================================

Weights of the model are automatically downloaded at :code:`./libreface_weights/` directory. If you want to download and save the weights to a separate directory, please specify the parent folder for weights as follows,

.. code-block:: python

    import libreface 
    libreface.get_facial_attributes(image_or_video_path,
                                    weights_download_dir = "your/directory/path")

Contributing
============

We welcome contributions! Here’s how you can help:

1. Fork the GitHub repository_.
2. Create a new branch for your feature (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

.. _repository: https://github.com/ihp-lab/LibreFace

License
=======
Please refer to our github repo for License_

.. _license : https://github.com/ihp-lab/LibreFace/blob/main/LICENSE.txt