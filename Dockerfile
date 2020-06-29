FROM python:3.8.3-buster

RUN mkdir /app

WORKDIR /app

RUN \
     git clone https://github.com/minillinim/GroopM.git \
  && cd /app/GroopM/ \
  && python3 setup.py install

CMD ["groopm"]
