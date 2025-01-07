FROM python:3.9
WORKDIR /usr/src/app
COPY ./src ./
CMD ["python", "./test.py"]