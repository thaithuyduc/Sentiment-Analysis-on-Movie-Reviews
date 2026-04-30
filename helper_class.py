from typing import Literal
import torch
import copy

class ModelTraining:
    def __init__(self, model, optimizer, loss_function, device,
                 train_loader, val_loader, test_loader):
        self.model = model
        self.optim = optimizer
        self.loss_fn = loss_function
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.best_state = None

    def calculate_loss_metric_score(self, loader, state: Literal['Train', 'Val', 'Test']):
        """
        Calculate loss and store values used to measure model's performance for each epoch

        Return
            Total loss for each epoch
            Lists of values (for each epoch) used to measure model's performance
        """
        total_loss = 0.0
        total_samples = 0
        y_label_pred = []
        y_label_true = []
        
        for X, y in loader:
            y = y.to(self.device)
            X = X.to(self.device)

            output = self.model(X) # (B, 3)

            loss = self.loss_fn(output, y)

            total_loss += loss.item() * len(y)
            total_samples += len(y)

            y_label_true.extend(y.tolist())
            y_label_pred.extend(torch.argmax(output, dim=1).tolist())

            if state == 'Train':
                # reset gradients to avoid gradient accumulation
                self.optim.zero_grad() 
                
                # calculate gradients
                loss.backward()
                
                # update weights
                self.optim.step()

        total_loss /= total_samples

        return y_label_true, y_label_pred, total_loss
    
    def train_performance(self, metric_fn: callable, kwargs: dict):
        """
        Measure performance of model based on specific metric in training for each epoch
        """
        self.model.train()
        y_label_true, y_label_pred, total_loss = self.calculate_loss_metric_score(self.train_loader, 'Train')
        metric_score = metric_fn(y_label_true, y_label_pred, **kwargs)
        
        return total_loss, metric_score

    def val_performance(self, metric_fn: callable, kwargs: dict):
        self.model.eval()
        with torch.no_grad():
            y_label_true, y_label_pred, total_loss = self.calculate_loss_metric_score(self.val_loader, 'Val')
            metric_score = metric_fn(y_label_true, y_label_pred, **kwargs)

            return total_loss, metric_score

    def training_session(self, epochs: int, metric_fn: callable, kwargs: dict|None = None):
        best_score = float('-inf')
        
        train_loss_collection = []
        val_loss_collection = []
        train_score_collection = []
        val_score_collection = []
        
        if kwargs is None:
            kwargs = {}
            
        for epoch in range(1, epochs + 1):
            train_loss, train_score = self.train_performance(metric_fn, kwargs)
            val_loss, val_score = self.val_performance(metric_fn, kwargs)
            
            train_loss_collection.append(train_loss)
            val_loss_collection.append(val_loss)

            train_score_collection.append(train_score)
            val_score_collection.append(val_score)
            
            if val_score > best_score:
                best_score = val_score
                self.best_state = copy.deepcopy(self.model.state_dict())
                print(f"New best score in epoch {epoch}: {val_score}. Saving model parameters...")

            print("=====================")
            print(f"Epoch {epoch} completed")
            print(f"Train Loss: {train_loss:.4f} ; Score: {train_score}")
            print(f"Val Loss: {val_loss:.4f} ; Score: {val_score}")
            print("=====================")
        
        return train_loss_collection, val_loss_collection, train_score_collection, val_score_collection, self.best_state
    
    def testing(self, metric_fn: callable, kwargs: dict|None=None):
        if kwargs is None:
            kwargs = {}
        
        self.model.load_state_dict(self.best_state)
        
        self.model.eval()
        with torch.no_grad():
            y_label_true, y_label_pred, test_loss = self.calculate_loss_metric_score(self.test_loader, 'Test')
            test_score = metric_fn(y_label_true, y_label_pred, **kwargs)

            print(f"Test Loss: {test_loss} ; Score: {test_score}")
        
        return test_loss, test_score