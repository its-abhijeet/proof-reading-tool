# **PROOFREADING TOOL**

The Proofreading Tool is a powerful tool that combines computer vision and natural language processing techniques to assist in proofreading and correcting grammar errors in scanned or digitally captured documents. It utilizes state-of-the-art technologies such as YOLO for text detection, Tesseract for optical character recognition (OCR), and a T5 model for grammar error correction.

## Introduction
Proofreading and correcting grammar errors in large documents or images can be a time-consuming and error-prone task. The Proofreading Tool aims to automate this process by leveraging advanced machine learning algorithms and techniques. By integrating text detection, OCR, and grammar error correction into a single tool, it provides an efficient and accurate solution for proofreading tasks.

## Features
### Text Detection
The Proofreading Tool employs the YOLO (You Only Look Once) algorithm for text detection. YOLO is a popular object detection algorithm known for its real-time performance and accuracy. It can detect text regions in images or documents and provide precise bounding box coordinates for each detected text region. This feature ensures that the tool accurately identifies and extracts the text areas from the input documents.

### Optical Character Recognition (OCR)
Once the text regions are identified, the Proofreading Tool utilizes Tesseract, a widely-used OCR engine. Tesseract converts the text regions into machine-readable text, enabling further processing and analysis. This OCR step ensures that the text within the detected regions is accurately extracted and prepared for the grammar error correction task.

### Grammar Error Correction
The extracted text obtained through OCR is passed through a fine-tuned T5 model specifically trained for the task of grammar error correction. T5 is a powerful transformer-based model known for its text generation capabilities. The T5 model carefully examines the extracted text, identifies grammar errors, and suggests corrections for each error. By utilizing a state-of-the-art language model, the Proofreading Tool provides accurate and contextually-aware grammar error correction suggestions.

These features combined make the Proofreading Tool an invaluable asset for proofreaders, content editors, and anyone who needs to ensure grammatical accuracy in their written materials. By automating the detection, extraction, and correction processes, the tool saves time and effort while improving the quality of written content.


# Installation

Clone the repository:


git clone https://github.com/eshangujar/proofreading-tool.git
Install the required dependencies:


pip install -r requirements.txt

The YOLO weights for text detection can be downloaded from darknet: https://pjreddie.com/darknet/yolo/.

The COCO-Text V2.0 dataset for custom training yolov3 can be downloaded from here: https://www.kaggle.com/datasets/c7934597/cocotext-v20
The C4_200M dataset for Grammar Error can be downloaded from here: https://www.kaggle.com/datasets/a0155991rliwei/c4-200m




Contributing
Contributions to the Proofreading Tool are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.




