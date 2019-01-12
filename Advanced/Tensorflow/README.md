# Tensor Flow

## Anaconda

**Anaconda** is a *free* and *open-source* distribution of the Python and R programming languages for scientific computing (data science, machine learning applications, large-scale data processing, predictive analytics, etc.), that aims to simplify package management and deployment.

Package versions are managed by the **package management** system ``conda``. The Anaconda distribution is used by over 6 million users and includes more than 1400 popular data-science packages suitable for Windows, Linux, and MacOS.

### Installation

- **Download** installer from [Anaconda Website](https://www.anaconda.com/download/).
- **Install** distribution depending on the O.S. distribution (Windows 10, MacOS or Linux).
- Set **Path** environment variables.

        i.e. PATH=&PATH;C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts

- **Verify** the installation

        python --version
        conda --version

- **Update** Conda version and packages

        conda update conda

  > Update without installing again the entire package

- Get the list of the *pre-installed* packages

        conda list

- Disable conda **ssl** verification

        conda config --set ssl_verify no

### Environments

Environments are useful to maintain multiple python and packages versions installed independently.

- To get the list with all environments use the following command:

        conda env list
        # or
        conda info --envs

   > The environment with the wildcard (``*``) is the current active one

- Create new anaconda (python) **environment**.

    > *SSLError*: to solve this issue, it must be activated firstly the default environment using the command: ``conda activate base``, then create the new environment. In this case the *SSLError* is not triggered anymore.

        # Get available python versions
        conda search "^python$"

        # Create empty conda environment using python version 3.6.7
        conda create -n tensorflow python=3.6.7

        # Create conda environment by using an existing one as a base
        conda create --name tensorflow --clone base
        conda create -n tensorflow --clone="C:\ProgramData\Anaconda3"

    > New environments are created inside `C:\ProgramData\Anaconda3\envs\`

- **Activate** current environment ``tensorflow``

        activate tensorflow
        #or
        conda activate tensorflow

- **Deactivate** an environment

        deactivate
        #or
        conda deactivate tensorflow

- To **verify** current environment it can be used the python command version (``python --version``) or use python code from *jupyter notebook* or *VSCode*, in order to verify the current **active** environment.

  ```python
  import sys
  print(sys.version)
  ```

        3.6.7 |Anaconda, Inc.| (default, Dec 10 2018, 20:35:02) [MSC v.1915 64 bit (AMD64)]

- **Update** packages fpr current environment

        python -m pip install --upgrade pip

- **Remove** an environment

        conda remove --name tensorflow --all

### Tensorflow

**TensorFlow** is an *open-source*e software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google

Tensorflow package will install other dependencies needed such as *numpy*, *protobuf*, *scipy*, *tensorboard*, *zlib*, etc..

- Install *current* Tensorflow release (CPU-only)

        # Install the package globally
        conda install -c anaconda tensorflow

        # Install the package for environment-only
        conda install -n tensorflow tensorflow

- Install Tensorflow (GPU package for CUDA-enabled GPU cards)

        conda install -n tensorflow tensorflow-gpu

- Install other packages such as: pandas, jupyter notebooks, pickle, matplotlib, scikit-learn etc..

    conda install -n tensorflow pandas jupyter matplotlib scikit-learn

#### Tensorboard

In order to open *tensorboard*, it is necessary to perform a call to the method ``tf.summary.FileWriter``, inside the script running the model

        writer = tf.summary.FileWriter(summaries_dir + '/train', session.graph)

Run the following command to start a tensorboard server, by locating path used for the logs. ``logs/train``

        tensorboard --logdir=logs/train

Use the [URL](http://JSANTOS-LAPTOP:6006) provided to enter into the dashboard.

### Jupyter notebooks

**Project Jupyter** is a nonprofit organization created to "*develop open-source software, open-standards, and services for interactive computing across dozens of programming languages*". Spun-off from IPython in 2014 by Fernando Pérez, Project Jupyter supports execution environments in several dozen languages. Project Jupyter's name is a reference to the three core programming languages supported by Jupyter, which are Julia, Python and R, and also an homage to Galileo's notebooks recording the discovery of the moons of Jupiter. 

Project Jupyter has developed and supported the interactive computing products Jupyter Notebook, Jupyter Hub, and Jupyter Lab, the next-generation version of Jupyter Notebook.

- To execute jupyter notebook, be sure jupyter **dependencies** are already installed into current environment.

        conda install -n tensorflow jupyter

- **Activate** desired environment

        activate tensorflow

- **Run** jupyter notebook server

        jupyter notebook

- Use the following [Jupyter Notebook URL](http://localhost:8888/?token=295ef2f127bc5ecdd1797500a820fd3e229962e135362d21) provided and create new documents using python or any language supported.

- To verify the current environment is used, used the following snippet to check the python version or any other library.

  ```python
  import sys
  print(sys.version)
  ```

## NVidia Graphics

Use NVidia GPU hardware graphics that supports CUDA technology to accelerate the computing using Tensorflow.

### Pre-requisites

[NVidia Tensorflow guide](https://www.quantinsti.com/blog/install-tensorflow-gpu)

The following NVIDIA® software must be installed on your system:

- NVIDIA® GPU drivers —CUDA 9.0 requires 384.x or higher.
- [CUDA® Toolkit](https://developer.nvidia.com/cuda-90-download-archive) —TensorFlow supports CUDA 9.0.
- CUPTI ships with the CUDA Toolkit.
- [cuDNN SDK](https://developer.nvidia.com/cudnn) (>= 7.2)
- (Optional) NCCL 2.2 for multiple GPU support.
- (Optional) TensorRT 4.0 to improve latency and throughput for inference on some models.

Add the following two paths to the path variable:

- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib

### Verify GPU enabled

Use the following python snippet to check if the GPU device is being recognized by tensorflow.

```python
import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    devices = sess.list_devices()
    print (devices)
    print (sess.run(c))
```

The console output must be similiar to the following lines. Check ``physical GPU`` is recognized and match with your current NVidia device.

```txt
Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1356 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1
2018-12-30 10:09:07.745313: I tensorflow/core/common_runtime/direct_session.cc:307] Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1

MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-30 10:09:07.768039: I tensorflow/core/common_runtime/placer.cc:927] MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
a: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-30 10:09:07.775052: I tensorflow/core/common_runtime/placer.cc:927] a: (Const)/job:localhost/replica:0/task:0/device:GPU:0
b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-30 10:09:07.781855: I tensorflow/core/common_runtime/placer.cc:927] b: (Const)/job:localhost/replica:0/task:0/device:GPU:0
```