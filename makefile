poly:
	wget "https://docs.google.com/uc?id=0B5lWReQPSvmGVEFzUEVzdTFBM0U&export=download" -O ./data/polyglot-zh.pkl

run:
	python build_data.py
	python train.py
	python evaluate.py
