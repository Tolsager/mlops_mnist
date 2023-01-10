FROM python:3.9-slim
run apt update && \
	apt install --no-install-recommends -y build-essential gcc && \
	apt clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY reports/ reports/
COPY models/ models/
RUN pip install -r requirements.txt --no-cache-dir
WORKDIR /
ENTRYPOINT ["python", "-u", "src/models/train_model.py", "train", "--device", "cpu"]
