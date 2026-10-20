install:  ; pip install -r requirements.txt
data:     ; python -m src.data_generation
train:    ; python -m src.train
report:   ; python -m src.evaluate
serve:    ; uvicorn app.main:app --reload
test:     ; pytest -q
all: data train report test
