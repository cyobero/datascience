FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
COPY . ./datascience
COPY ./custom.css ~/.jupyter/custom/custom.css
EXPOSE 8888

USER root

RUN pip install tensorflow-datasets \
	nltk \
	tflearn \
	sklearn \
	pandas
