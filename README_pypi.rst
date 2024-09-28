=====================================
Libreface
=====================================

`GitHub Repo`_ - https://github.com/ihp-lab/LibreFace

.. _`GitHub Repo`: https://github.com/ihp-lab/LibreFace

|badge1| |badge2|


.. |badge1| image:: https://img.shields.io/badge/version-1.0.0-blue
   :alt: Static Badge


.. |badge2| image:: https://img.shields.io/badge/python-%3D%3D3.8-green
   :alt: Static Badge


This is the Python package for `LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis`_.
LibreFace is an open-source and comprehensive toolkit for accurate and real-time facial expression analysis with both CPU and GPU acceleration versions.
LibreFace eliminates the gap between cutting-edge research and an easy and free-to-use non-commercial toolbox. We propose to adaptively pre-train the vision encoders with various face datasets and then distill them to a lightweight ResNet-18 model in a feature-wise matching manner.
We conduct extensive experiments of pre-training and distillation to demonstrate that our proposed pipeline achieves comparable results to state-of-the-art works while maintaining real-time efficiency.

.. _`LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis`: https://boese0601.github.io/libreface



Dependencies
============

- Python 3.8
- You should have `cmake` installed in your system.
    - **For Linux users** - :code:`sudo apt-get install cmake`. If you run into trouble, consider upgrading to the latest version (`instructions`_).
    - **For Mac users** - :code:`brew install cmake`.

.. _`instructions`: https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line


Installation
============
You can first create a new Python 3.8 environment using `conda` and then install this package using `pip` from the PyPI hub:

.. code-block:: bash
    
    conda create -n libreface_env python=3.8
    conda activate libreface_env
    pip install --upgrade libreface


Usage
=====

Commandline
----------------

You can use this package through the command line using the following command.

.. code-block:: bash

    libreface --input_path="path/to/your_image_or_video"

Note that the above script would save results in a CSV at the default location - :code:`sample_results.csv`. If you want to specify your own path, use the :code:`--output_path`  command line argument,

.. code-block:: bash

    libreface --input_path="path/to/your_image_or_video" --output_path="path/to/save_results.csv"

To change the device used for inference, use the :code:`--device` command line argument,

.. code-block:: bash

    libreface --input_path="path/to/your_image_or_video" --device="cuda:0"

To save intermediate files, :code:`libreface` uses a temporary directory that defaults to ./tmp. To change the temporary directory path,

.. code-block:: bash

    libreface --input_path="path/to/your_image_or_video" --temp="your/temp/path"

For video inference, our code processes the frames of your video in batches. You can specify the batch size and the number of workers for data loading as follows,

.. code-block:: bash

    libreface --input_path="path/to/your_video" --batch_size=256 --num_workers=2 --device="cuda:0"

Note that by default, the :code:`--batch_size` argument is 256, and :code:`--num_workers` argument is 2. You can increase or decrease these values according to your machine's capacity.

**Examples**

Download a `sample image`_ from our GitHub repository. To get the facial attributes for this image and save to a CSV file, simply run,

.. _`sample image`: https://github.com/ihp-lab/LibreFace/blob/pypi_wrap/sample_disfa.png

.. code-block:: bash

    libreface --input_path="sample_disfa.png"

Download a `sample video`_ from our GitHub repository. To run the inference on this video using a GPU and save the results to :code:`my_custom_file.csv` run the following command,

.. _`sample video`: https://github.com/ihp-lab/LibreFace/blob/pypi_wrap/sample_disfa.avi

.. code-block:: bash
    
    libreface --input_path="sample_disfa.avi" --output_path="my_custom_file.csv" --device="cuda:0"

Note that for videos, each row in the saved CSV file corresponds to individual frames in the given video.

Python API
--------------

Here’s how to use this package in your Python scripts.

To assign the results to a Python variable,

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

For video inference, our code processes the frames of your video in batches. You can specify the batch size and the number of workers for data loading as follows, 

.. code-block:: python

    import libreface 
    libreface.get_facial_attributes(video_path,
                                    batch_size = 256,
                                    num_workers = 2)

Note that by default, the :code:`batch_size` is 256, and :code:`num_workers` is 2. You can increase or decrease these values according to your machine's capacity.

Downloading Model Weights
================================

Weights of the model are automatically downloaded at :code:`./libreface_weights/` directory. If you want to download and save the weights to a separate directory, please specify the parent folder for weights as follows,

.. code-block:: python

    import libreface 
    libreface.get_facial_attributes(image_or_video_path,
                                    weights_download_dir = "your/directory/path")

Output Format
==================

For an image processed through LibreFace, we save the following information in the CSV file,

- :code:`lm_mp_idx_x`, :code:`lm_mp_idx_y`, :code:`lm_mp_idx_z` :  x, y, z co-ordinate of the 3D landmark indexed at :code:`idx` (total 478) obtained from mediapipe. Refer to the `mediapipe documentation`_ for getting the index to landmark map.

- :code:`pitch`, :code:`yaw`, :code:`roll` : contains the angles in degrees for the 3D head pose for the person.

- :code:`facial_expression` : contains the detected facial expression. Can be "Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", or "Contempt".

- :code:`au_idx` : contains the output of our action unit (AU) detection model, which predicts whether an action unit at index :code:`idx` is activated. 0 means not activated, and 1 means activated. We detect AU at the indices :code:`[1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]`.

- :code:`au_idx_intensity` : contains the output of our action unit (AU) intensity prediction model, which predicts the intensity of an action unit at index :code:`idx` between 0 and 5. 0 is least intensity and 5 is maximum intensity. We predict AU intensities for the AU indices :code:`[1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]`.

.. _`mediapipe documentation`: https://github.com/google-ai-edge/mediapipe/blob/7c28c5d58ffbcb72043cbe8c9cc32b40aaebac41/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

For a video, we save the same features for each frame in the video at index :code:`frame_idx` and timestamp :code:`frame_time_in_ms`.

Inference Speed
====================

LibreFace is able to process long-form videos at :code:`~30 FPS`, on a machine that has a :code:`13th Gen Intel Core i9-13900K` CPU and a :code:`NVIDIA GeForce RTX 3080` GPU. Please note that the default code runs on CPU and you have to use the :code:`device` parameter for Python or the :code:`--device` command line option to specify your GPU device ("cuda:0", "cuda:1", ...).

Contributing
============

We welcome contributions! Here’s how you can help:

1. Fork the GitHub repository_.
2. Create a new branch for your feature (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

.. _repository: https://github.com/ihp-lab/LibreFace

If you have questions about this package, please direct them to achaubey@usc.edu or achaubey@ict.usc.edu

License
=======
Please refer to our github repo for License_

.. _license : https://github.com/ihp-lab/LibreFace/blob/main/LICENSE.txt