"""
This file contains code to train models.
"""


import os
import numpy as np
import torch
from qumia_dataset import QUMIA_Dataset
from qumia_confusion import create_confusion_matrix
import wandb


class QUMIA_Trainer:
    """ Object to hold the shared objects to be accessed during training and validation.
        (So we're not passing around a bunch of arguments to functions all the time.)
    """

    def __init__(self, df_train, df_validation, train_loader, validation_loader, device, model, criterion, optimizer, output_dir):
        self.df_train = df_train
        self.df_validation = df_validation
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.output_dir = output_dir


def train(num_epochs, trainer: QUMIA_Trainer):

    train_loader = trainer.train_loader
    validation_loader = trainer.validation_loader
    device = trainer.device
    model = trainer.model
    criterion = trainer.criterion
    optimizer = trainer.optimizer
    output_dir = trainer.output_dir

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0

        n_seconds = 10  # Print every n seconds
        for i, data in enumerate(train_loader, 0):
            inputs = data['image']
            labels = data['label']
            fuse_features = data['fuse_features']
            
            # Reshape labels to match output of model
            labels = labels.view(-1, 1).float()

            # Move input and label tensors to the default device
            inputs = inputs.to(device)
            labels = labels.to(device)
            fuse_features = fuse_features.to(device)
            
            # print the shape of the input and label tensors
            #print(inputs.shape, labels.shape)
            #print(inputs.dtype, labels.dtype)

            optimizer.zero_grad()

            outputs = model(inputs, fuse_features)
            # print(outputs.shape)
            # print(outputs.dtype)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch}.pth'))

        _, _, train_loss = make_predictions(trainer, train_loader)
        print(f"Train loss: {train_loss:.4f}")
        _, _, validation_loss = make_predictions(trainer, validation_loader)
        print(f"Validation loss: {validation_loss:.4f}")

        wandb.log({"train-loss": train_loss, "validation-loss": validation_loss, "epoch": epoch})

    # Save the model and weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

    # Do the final validation run (saving the predictions)
    validate(trainer, set_type='validation')
    validate(trainer, set_type='train')

    wandb.finish()



def validate(trainer: QUMIA_Trainer, n_batches=None, set_type='validation'):
    """ This will evaluate the model on the validation or train dataset (set_type),
        save the predictions to a csv file and generate a confusion matrix.
    """
    assert set_type in ['validation', 'train']
    loader = trainer.validation_loader if set_type == 'validation' else trainer.train_loader
    df = trainer.df_validation if set_type == 'validation' else trainer.df_train

    # Make predictions on the specified dataset
    predictions, labels, loss = make_predictions(trainer, loader, n_batches)
    print(f"{set_type} loss: {loss:.4f}")
    
    # Convert predictions and labels to numpy arrays, and map back to original h_score values
    predictions = predictions.cpu().numpy().flatten()
    predictions = np.array(predictions, dtype=np.float32)
    rounded_predictions = np.round(predictions)
    labels = labels.cpu().numpy().flatten()
    labels = np.array(labels, dtype=np.float32)
    print(predictions.shape, labels.shape)
    print(rounded_predictions.dtype)

    # We might only have predictions for a number of batches, so we need to trim the dataframe
    df_combined = df.iloc[:predictions.shape[0]].copy()

    # Combine the original dataframe with the predictions
    df_combined['prediction'] = predictions
    df_combined['rounded_prediction'] = rounded_predictions
    df_combined['label'] = labels # redundant, but we could detect a mismatch with the inputs maybe

    # As a sanity check, see if the labels match the original input rows
    match = df_combined['label'].equals(df_combined['h_score'].astype('float32'))
    print(f"Labels match: {match}")
    if not match:
        print("Possible mismatch between labels and inputs!")
        #raise Exception("Mismatch between labels and inputs")

    # Save the dataframe to a csv file
    df_combined.to_csv(os.path.join(trainer.output_dir, f'df_{set_type}_predictions.csv'), index=False)

    create_confusion_matrix(rounded_predictions.tolist(), labels.tolist(), set_type, trainer.output_dir)

    return df_combined



def make_predictions(trainer: QUMIA_Trainer, dataloader, n_batches=None):
    """ Makes predictions on the given dataloader (train / validation / test data) using the given model.
        It will return the predictions and the ground-truth labels.
    """
    trainer.model.eval()  # Set the model to evaluation mode

    predictions = []
    labels = []
    loss = None

    with torch.no_grad():
        running_loss = 0.0
        for index, batch in enumerate(dataloader, 0): # tqdm(dataloader, total=len(dataloader), desc="Performing predictions on validation data"):
            inputs = batch['image']
            batch_labels = batch['label'].view(-1, 1).float()
            fuse_features = batch['fuse_features']

            # Move input and label tensors to the default device
            inputs = inputs.to(trainer.device)
            batch_labels = batch_labels.to(trainer.device)
            fuse_features = fuse_features.to(trainer.device)

            # Forward pass
            outputs = trainer.model(inputs, fuse_features)
            
            # Save predictions and labels
            predictions.append(outputs)
            labels.append(batch_labels)

            # Compute loss
            loss = trainer.criterion(outputs, batch_labels)
            running_loss += loss.item()

            index += 1
            if n_batches is not None and index > n_batches:
                break

        loss = running_loss / len(dataloader)

    return torch.cat(predictions), torch.cat(labels), loss



def get_weighted_loss_function(class_weights):

    def weighted_mse_loss(input, target):
        """ Weighted MSE loss function.
            This function is used to calculate the loss of the model.
        """
        assert input.shape == target.shape, "Input and target must have the same shape"

        # Assign weights based on the target class
        # This assumes targets are 1.0, 2.0, 3.0, and 4.0 for the classes
        sample_weights = class_weights[target.long() - 1]

        # Calculate MSE loss for each sample
        mse = torch.nn.functional.mse_loss(input, target, reduction='none')

        # Weight the MSE loss by the sample weights
        weighted_mse = mse * sample_weights

        # Return the mean loss
        return weighted_mse.mean()

    return weighted_mse_loss