FROM tensorflow/tensorflow:2.11.0-gpu
WORKDIR /app
COPY ./check ./check
COPY ./datasets ./datasets
COPY ./estimators ./estimators
COPY ./utils ./utils
COPY ./experiments ./experiments
COPY ./requirements.txt .
COPY ./experimenter_kfold.py .
COPY ./experimenter_cross_dataset.py .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt -v
# CMD [ "python", "experimenter_kfold.py" ]