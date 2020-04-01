
.PHONY: data train predict

train: data models/food_classifier.pt
	./source/train_model.py

predict: data models/prediction.pt
	./source/predict_model.py

data: data/images
	./source/data/make_dataset.py

