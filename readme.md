## Experiment: CNN - Computer Vision 

### custom setting
- pyton env: 3. 9
- paste config.yaml and custom setting 
- change load_config path in runner.ipynb 
- if using window, 
    1. change module code line in runner.ipynb
        !python -m pip install --upgrade pip
        !python -m pip install pyyaml pandas numpy matplotlib seaborn scikit-learn
        !python -m pip uninstall keras -y
        !python -m pip install "tensorflow>=2.16"
        !python -m pip install opencv-python  
    2. change function name on load_config cell: load_config > load_config_win_env
    3. write the path with '/'

### model 
model build
- ResNet50, MobileNetV2, EfficientNetB0, DenseNet121
    - imagenet weights model : pre-trained
- backbone > global average pooling 2D > dense(relu, dropout) > outputs dense(softmax)
- data agumentation:
    rotation_range 20
    width_shift_range 0.1
    height_shift_range 0.1
    zoom_range 0.2
    horizontal_flip
    vertical_flip
    - additional: Color agumentation(color_agu.py)

optimizer
- adamw
- weight decay
- dropout

callback
- early stopping
- model checkpoint
- learning schedular: reduce learning rate on plateau

output layer
- Dense layer
- activation: softmax