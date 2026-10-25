CAT ?= bottle
.PHONY: install test train evaluate baseline api dashboard bench all-categories
install:
	pip install -r requirements-dev.txt
test:
	pytest -q
train:
	python -m src.train --set data.category=$(CAT)
baseline:
	python -m src.train --set data.category=$(CAT) model.name=global_knn model.knn_k=3 model.backbone=resnet18
	python -m src.evaluate --set data.category=$(CAT) model.name=global_knn
evaluate:
	python -m src.evaluate --set data.category=$(CAT)
api:
	MODEL_DIR=models/patchcore/$(CAT) uvicorn app.api:app --reload
dashboard:
	streamlit run dashboard/streamlit_app.py
bench:
	python -m src.benchmark --model-dir models/patchcore/$(CAT) \
	  --image $$(ls data/mvtec_ad/$(CAT)/test/good/*.png | head -1)
