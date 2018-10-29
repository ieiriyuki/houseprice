FROM python:3.6.6

WORKDIR /work

RUN pip install --upgrade pip \
    && pip install \
        numpy==1.14 \
        matplotlib==2.2 \
        pandas==0.22 \
        seaborn==0.8 \
        scipy==1.1 \
        scikit-learn==0.19 \
        pandas-profiling==1.4
