DATA_DIR = data

DATA_RAW_DIR = $(DATA_DIR)/raw
DATA_MNIST_DIR = $(DATA_DIR)/tutorial_04/MNIST/raw

URL_CATS_AND_DOGS =  https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip
FILE_CATS_AND_DOGS = $(DATA_RAW_DIR)/kagglecatsanddogs_5340.zip

all: download

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(DATA_RAW_DIR):
	mkdir -p $(DATA_RAW_DIR)

$(DATA_MNIST_DIR):
	mkdir -p $(DATA_MNIST_DIR)

cats_and_dogs: $(DATA_RAW_DIR)
	wget -O $(FILE_CATS_AND_DOGS) $(URL_CATS_AND_DOGS)
	unzip -o $(FILE_CATS_AND_DOGS) -d $(DATA_RAW_DIR)

mnist: $(DATA_MNIST_DIR)
	wget -O $(DATA_MNIST_DIR)/train-images-idx3-ubyte.gz https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
	wget -O $(DATA_MNIST_DIR)/train-labels-idx1-ubyte.gz https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
	wget -O $(DATA_MNIST_DIR)/t10k-images-idx3-ubyte.gz https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
	wget -O $(DATA_MNIST_DIR)/t10k-labels-idx1-ubyte.gz https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
	gunzip -f $(DATA_MNIST_DIR)/*.gz

download: cats_and_dogs mnist

test:
	pytest tests/ -vv -s
