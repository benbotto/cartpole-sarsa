FROM python:3-bookworm

RUN pip install "gymnasium[classic-control]==1.2.0"

WORKDIR /app

ENTRYPOINT [ "python" ]
CMD [ "cartpole-sarsa.py" ]
