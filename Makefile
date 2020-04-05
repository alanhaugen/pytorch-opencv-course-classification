
.PHONY: train

train: data/images models/food_classifier.pt
	./trainer/train_model.py

models/food_classifier.pt: data/images
	./trainer/predict_model.py

data/images:
	./trainer/data/make_dataset.py

