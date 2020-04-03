
.PHONY: train

train: data/images models/food_classifier.pt
	./source/train_model.py

models/food_classifier.pt: data/images
	./source/predict_model.py

data/images:
	./source/data/make_dataset.py

