# Technical Skills
### Computer Vision, Natural Language Processing, Machine Learning, Artificial IntelligenceC++, Python, C, C#, Java, Pytorch, TensorFlow, Kali Linux, HTML5, CSS, JavaScript, PHP, jQuery, AJAX, SQL, Firebase, XML, Accela, Crystal Reports, Toad SQL Server

# Research
**Comparative Study of Facial Detection Algorithms**
- Researched Haars Cascade, MTCNN, Dlib, DNN, RetinaFace, and HOG facial detection algorithms in efforts to compare their performances.
- Implemented all algorithms in three mediums: photos, videos, and live-stream videos.
- Goal: To determine the performance of currently used facial detection algorithms, highlighting the benefits and limitations of each in different situations
- [Research](https://github.com/mbcruz96/Facial-detection)

**Clinical Report Generator**
- Designed a Transformer model with the following architecture: a multimodal CNN encoder and a generative LSTM decoder to generate clinical reports based off individual patient diagnostic information.
- After training multiple models, the model weights were averaged using Federated Learning to create a global model and propagated the weights back to each individual model.
- By utilizing a federated learning, the model could be upscaled so that hospitals could use the model to generate clinical reports without sharing private patient information across hospitals.
- Goal: To implement a model that would allow medical data to be used in deep learning models without breaching constraints imposed by HIPPA
- [Research](https://github.com/mbcruz96/Clinical-Report-Generation.git)

**Optimal Hardware Implementation of Multi-Level Cache Based on Software Simulation**
- Research centered around designing various multi-level cache software implementations with various architectures and measuring each implementation on how it impacts power consumption, timing, and utilization on a benchmark FPGA
- The cache simulators determined the optimal miss rate which was implemented in hardware via Verilog, SystemVerilog, and Vivado
- Goal: Determine the hardware benefits and drawbacks of high-performance multi-level cache designs as well as finding the optimal design architecture 
  
# Projects
**Canny Edge Detector**
- Implemented the Canny Edge Detector
- Images are smoothed using a Gaussian filter and then their orientations and magnitudes are calculated.
- Non-maximum suppression is performed on the image pixels to discover true edge pixels.
- Using hysterisis thresholding, dominant edge pixels are discovered using eight way connectivity. 
- [Code](https://github.com/mbcruz96/Canny-Edge-Detection.git)

**Digit AutoEncoder**
- Using the MNIST dataset, an autoencoder model was implemented that predicts what digit a photo contains
- The model encodes the image into a latent space and decodes a representation of the original image with upsampling
- [Code](https://github.com/mbcruz96/AutoEncoder)

**Otsu Thresholding**
- Implemented the Otsu thresholding algorithm for binary image segmantation
- This implementation iteratively uses each pixel intensity from 0-255 to calculate the intra class variance between the two distributions created by using the current pixel intensity as the seperating boundary of the distributions.
- The intensity which yields the highest variance amongst the distributions is chosen as the thresholding value used to alter the original image.
- [Code](https://github.com/mbcruz96/Otsu)

**Generative Shakespearian Speech Model**
- NLP model that generates text in the style of Shakesperian english
- The model is pretrained on a corpus containing some of Shakespear's work
- Using a seed, the model generatively creates sentances reminiscent of old English vernacular
- [Code](https://github.com/mbcruz96/LSTM/blob/main/TextGenerator.ipynb)
  
**MCTL template library**
- Created the Michael Cruz template library, which implements some of the data structures available in the C++ std library.
- [Code](https://github.com/mbcruz96/MCTL.git)

  
  
