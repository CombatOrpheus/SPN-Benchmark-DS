FROM python:3.9

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install torch torchvision
RUN pip install joblib
RUN pip install tqdm
RUN pip install jsonschema jsonpickle
RUN pip install tensorboard tensorboardX
RUN pip install scikit-learn
RUN pip install xlrd xlutils xlwt
RUN pip install dgl
RUN pip install scipy
RUN pip install torch-scatter

CMD ["pip", "list"]

