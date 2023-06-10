FROM nvcr.io/nvidia/pytorch:23.05-py3
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple