FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi

RUN pip install -r requirements.txt --default-timeout=100
RUN git clone https://github.com/cahya-wirawan/WikiSearch.git

COPY start.sh /fastapi

EXPOSE 8000

RUN chmod +x ./start.sh

CMD ["./start.sh"]
