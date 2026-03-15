DATA_DIR = data/raw
URL =  https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip
FILE = $(DATA_DIR)/kagglecatsanddogs_5340.zip

all: download unzip

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

download: $(DATA_DIR)
	wget -O $(FILE) $(URL)

unzip: download
	unzip -o $(FILE) -d $(DATA_DIR)

test:
	pytest tests/ -vv -s
