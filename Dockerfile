FROM python:3.8-slim-buster

WORKDIR /workspace

RUN apt-get update && apt-get install apt-file -y && apt-file update && apt-get install vim -y  && apt-get install git-lfs -y
RUN git config --global user.email "kunal.suri.ml.experiments@gmail.com"
RUN git config --global user.name "Kunal Suri"

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["jupyter","notebook","--ip=0.0.0.0","--no-browser","--allow-root"]
