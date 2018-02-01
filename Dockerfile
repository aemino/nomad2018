FROM tensorflow/tensorflow:1.4.0-gpu-py3

WORKDIR /srv

ADD requirements.txt /srv/

RUN pip3 --no-cache-dir install \
    -r /srv/requirements.txt

ADD . /srv

RUN chmod +x /srv/permute_net.py

ENV TF_MIN_GPU_MULTIPROCESSOR_COUNT 2

CMD ["/srv/permute_net.py"]

