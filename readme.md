## Experiment: CNN - Computer Vision 

### custom setting
- paste config.yaml and custom setting 
- change load_config path in runner.ipynb 
- if using window, change module code line in runner.ipynb

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