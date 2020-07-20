config = {
    # Basic details
    'num_classes': 91,
    'device' : 'cuda',
    'name': 'COCO', # experiment name

    'num_workers' : 4,

    # Prior Box features 
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'image_size': 300, 
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],

    # Train features
    'lr' : 1e-3,
    'batch_size' : 4,
    'weight_decay' : 0.01,
    'num_epochs' : 30, # max epochs to train
    'log_every_train' : 50, # how many batches after to print info regarding losses
    'log_every_val' : 50,

    'alpha' : 1.0,
    'epochs_lr' : [10, 20, 30, 40], # epochs after which to change lr
    'gamma' : 0.1, # factor by which to change
    'save_model' : True,
    'save_model_epochs' : [10, 20, 25, 30] # epochs after which to save model
}