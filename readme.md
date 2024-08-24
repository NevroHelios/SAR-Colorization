### Training the model

1. Create a virtual environment: `python -m venv sar_colorization`
2. Activate the virtual environment: `source bin/activate` (on Linux/Mac) or `.\sar_colorization\Scripts\activate` (on Windows)
3. change dir `cd SAR-Colorization`
4. Install the requirements: `pip install -r requirements.txt`
5. Run the training script: `python train.py`
6. To check the model performance, run: `tensorboard --logdir=runs` and open `http://localhost:6006/` in your browser.
