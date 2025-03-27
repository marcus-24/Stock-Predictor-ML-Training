FROM tensorflow/serving:latest

RUN git lfs install

RUN git clone https://huggingface.co/DrMarcus24/test-stock-predictor 

ENV MODEL_NAME stock-pred-model

COPY "hf://DrMarcus24/test-stock-predictor" /models/$MODEL_NAME