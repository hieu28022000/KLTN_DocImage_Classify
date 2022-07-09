FROM ubuntu:20.04


# ==================================================================
# Setup ubuntu
# ------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    apt-get install -y \
        nano \
        cmake \
        wget


# ==================================================================
# Install require package
# ------------------------------------------------------------------
RUN apt-get install -y python3-opencv && \
    apt-get install -y tesseract-ocr

RUN pip install \
        pyyaml==5.4.1 \
        pandas==1.4.3 \
        easydict==1.9 \
        tensorflow==2.8.0 \
        tensorflow_addons==0.16.1 \
        tf-models-official==2.7.0 \
        tensorflow_text==2.8.2 \
        opencv-python==4.6.0.66 \
        pytesseract==0.3.9 \
        datasets==2.2.2 \
        transformers==4.19.4 \
        torch==1.11.0 \
        protobuf==3.20.0 \
        Flask==2.1.2
    

# ==================================================================
# Create work space
# ------------------------------------------------------------------
COPY . /server
WORKDIR /server
EXPOSE 80


# ==================================================================
# Run
# ------------------------------------------------------------------
CMD ["python", "classify_api.py"]