import torch
import matplotlib.pyplot as plt

class TrainingContainer:
    def __init__(train_dataloader, val_dataloader, test_dataloader,
                 model, loss_func, optimizer,
                 batch_size, learning_rate, epoch_number):
        self.training_data = train_dataloader
        self.val_data = val_dataloader
        self.test_data = test_dataloader
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch_number = epoch_number