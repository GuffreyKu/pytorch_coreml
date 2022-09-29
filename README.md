Environment:
1. Install Xcode : xcode-select --install
2. Install Homebrew : /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
3. Install miniforge : brew install miniforge
4. conda create -n torch python=3.10
5. conda activate torch-gpu

python tool :
1. pip install torch torchvision torchaudio
2. conda install pytorch-lightning -c conda-forge
3. pip install coremltools
4. conda install -c conda-forge opencv
5. pip install pandas
6. pip install tensorboard

how to training : 
1. download dataset : https://www.notion.so/https-www-kaggle-com-datasets-gpiosenka-100-bird-species-8b505698405c48c9ad7ab73a705742a3
2. create "data" folder and put in bird data.
3. python trainer_lightning.py

how to convert model
1. modify model path
2. python pt2ct.py

how to inference
1. python inference.py

how to inference on Iphone
1. use Xcode open "CoreMLDemo" folder
2. put in mlmodel in folder
3. compile it

good luck
