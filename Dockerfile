FROM minillinim/bamm:latest

RUN \
     python3 -m pip install --upgrade pip \
  && python3 -m pip install --upgrade setuptools

WORKDIR /app

RUN \
     git clone https://github.com/minillinim/GroopM.git \
  && cd /app/GroopM/ \
  && python3 setup.py install

CMD ["groopm"]
