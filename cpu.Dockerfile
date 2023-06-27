FROM python:3.10-alpine
COPY ./requirements.cpu.txt /requirements.txt
RUN pip install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple