# `pip3 install nextoff`
![Nextoff](docs/static/images/NEXTOFF.png)

![](https://aleen42.github.io/badges/src/tensorflow.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/rakesh4real/nextoff/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://twitter.com/_rakesh4real)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/rakesh4real/nextoff)


A package to perform emprical hyperparameter tuning!

# How to use
Everything revolves around args
```
class args:
    """
    BUILD MODEL CONFIGS:
    
        + input_shape {tuple}       : discluding batch size
        + dropout_p {Float}         : probability
        + conv_seq {list of dicts}  : see docstring
        + fcs_seq {list of dicts}   : see docstring
        
    
    """
    input_shape  = (32,32,3)
    dropout_p    = 0.5
    conv_seq     = [
        {"out_ch": 55, "z": 11, 'act': 'relu'},
        {"maxpool_z": 2},
        # -------------------------------------------
        {"out_ch": 27, "z": 5, 'act': 'relu'},
        {"maxpool_z": 2},
        # -------------------------------------------
        {"out_ch": 13, "z": 3, 'act': 'relu'},
        {"out_ch": 13, "z": 3, 'act': 'relu'},
        {"out_ch": 13, "z": 3, 'act': 'relu'},
        {"maxpool_z": 2},
        # -------------------------------------------
    ] 
    fcs_seq     = [
        {"out_nodes": 4096, "act":    'relu', 'p': 0.5 },
        {"out_nodes": 4096, "act":    'relu', 'p': 0.5 },        
        {"out_nodes":   10, "act": 'softmax', 'p': 0   },        
    ]
    
    # optimisation
    # ---------------------------
    # optimizer = "RMSprop"
    # rms_decay = 1e-6
    # lr        = 1e-4
    # ----------------------------
    optimizer    = "SGD"
    sgd_momentum = float(0.9)
    sgd_nesterov = False
    lr           = 1e-2
    # ----------------------------
    lr        = 1e-4
    loss      = 'categorical_crossentropy'
    metrics  = ['accuracy']
    
    # training
    batch_size = 64
    epochs = 1
    shuffle = True
    
    # ==============================================
    # custom callbacks
    # ==============================================
    """ uncomment to see in action
    class CustomCallback(keras.callbacks.Callback):
        def __init__(self, num_batches):
            super(args.CustomCallback, self).__init__()
            self.tot_batches = num_batches

        def on_train_batch_begin(self, batch, logs=None):
            keys = list(logs.keys())
            os.system('clear')
            print(f"status {batch/self.tot_batches}%")
            os.system('clear')
    """
```

----

[![Twitter](https://aleen42.github.io/badges/src/twitter.svg)](https://twitter.com/_rakesh4real)

Connect with the maintainer

----
**To contributers:** Build new release to update PyPI. Thanks to GitHub Actions!