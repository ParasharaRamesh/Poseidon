from models.simple_pose import SimplePose
from torch import nn
from torch.utils.data import DataLoader
from dataloader.h36M_loader import Human36MLoader
from tqdm import tqdm
import torch
from datetime import datetime
import argparse
import os
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def kaiming_weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        
def train_once(train_dict):
    model = train_dict['model']
    dataloader = train_dict['dataloader']
    device = train_dict['device']
    optimizer = train_dict['optimizer']
    three_dim_pose_loss_fn = train_dict['three_dim_pose_loss_fn']
    action_label_loss_fn = train_dict['action_label_loss_fn']
    
    predicted_labels = None
    true_labels = None
    total_losses = []
    pose_losses = []
    action_losses = []
    
    model.train()
    for data in tqdm(dataloader):
        # Prepare Data
        two_dim_input_data, three_dim_output_data, action_labels= data
        two_dim_input_data = two_dim_input_data.to(device)
        three_dim_output_data = three_dim_output_data.to(device)
        action_labels = action_labels.to(device)
        # Set Gradients to 0
        optimizer.zero_grad()
        # Train Model
        predicted_3d_pose_estimations, predicted_action_labels = model(two_dim_input_data)
        # Calculate Loss
        three_dim_pose_estimation_loss = three_dim_pose_loss_fn(predicted_3d_pose_estimations, three_dim_output_data)
        action_label_loss = action_label_loss_fn(predicted_action_labels, action_labels)
        loss = three_dim_pose_estimation_loss + action_label_loss
        # Store Results
        total_losses.append(loss)
        pose_losses.append(three_dim_pose_estimation_loss)
        action_losses.append(action_label_loss)
        predicted_action_labels = torch.argmax(predicted_action_labels, axis=1)
        predicted_labels = predicted_action_labels if predicted_labels is None else torch.cat((predicted_labels, predicted_action_labels), axis=0)
        true_labels = action_labels if true_labels is None else torch.cat((true_labels, action_labels), axis=0)
        # Optimize Gradients and Update Learning Rate
        loss.backward()
        optimizer.step()
    
    return predicted_labels, true_labels, total_losses, pose_losses, action_losses

def test_once(test_dict):
    model = test_dict['model']
    dataloader = test_dict['dataloader']
    device = test_dict['device']
    three_dim_pose_loss_fn = test_dict['three_dim_pose_loss_fn']
    action_label_loss_fn = test_dict['action_label_loss_fn']
    
    predicted_labels = None
    true_labels = None
    total_losses = []
    pose_losses = []
    action_losses = []
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            # Prepare Data
            two_dim_input_data, three_dim_output_data, action_labels= data
            two_dim_input_data = two_dim_input_data.to(device)
            three_dim_output_data = three_dim_output_data.to(device)
            action_labels = action_labels.to(device)
            # Predict with model
            predicted_3d_pose_estimations, predicted_action_labels = model(two_dim_input_data)
            # Calculate Loss
            three_dim_pose_estimation_loss = three_dim_pose_loss_fn(predicted_3d_pose_estimations, three_dim_output_data)
            action_label_loss = action_label_loss_fn(predicted_action_labels, action_labels)
            loss = three_dim_pose_estimation_loss + action_label_loss
            # Store Results
            total_losses.append(loss)
            pose_losses.append(three_dim_pose_estimation_loss)
            action_losses.append(action_label_loss)
            predicted_action_labels = torch.argmax(predicted_action_labels, axis=1)
            predicted_labels = predicted_action_labels if predicted_labels is None else torch.cat((predicted_labels, predicted_action_labels), axis=0)
            true_labels = action_labels if true_labels is None else torch.cat((true_labels, action_labels), axis=0)

    return predicted_labels, true_labels, total_losses, pose_losses, action_losses

def print_evaluation_metric(epoch, predicted_labels, true_labels, total_losses, pose_losses, action_losses, mode):
    correct_predictions = (predicted_labels == true_labels).sum().item()
    accuracy = correct_predictions / predicted_labels.size(0) * 100
    total_loss = sum(total_losses) / len(total_losses)
    pose_loss = sum(pose_losses) / len(pose_losses)
    action_loss = sum(action_losses) / len(action_losses)
    
    if mode == 'test':
        print(f"Epoch: {epoch} | Total Testing Loss: {total_loss} | Pose Testing Loss: {pose_loss} | Action Testing Loss: {action_loss} | Action Test Label Accuracy: {accuracy}")
    elif mode == 'train':
        print(f"Epoch: {epoch} | Total Training Loss: {total_loss} | Pose Training Loss: {pose_loss} | Action Training Loss: {action_loss} | Action Train Label Accuracy: {accuracy}")

def save_model(epoch, model, optimizer, scheduler):
    logging.info(f'Saving model current state')
    state_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    weight_save_path = os.path.join('weights', 'simple_pose')
    if not os.path.exists(weight_save_path):
        os.makedirs(weight_save_path)

    torch.save(state_dict, os.path.join(weight_save_path, f'weights_{epoch}.pth'))

def training_loop(args):
    SAVE_PATH = args.save_path
    TRAINING_2D_DATA_PATH = args.training_2d_data_path
    TRAINING_3D_DATA_PATH = args.training_3d_data_path
    TRAINING_LABEL_PATH = args.training_label_path
    TESTING_2D_DATA_PATH = args.testing_2d_data_path
    TESTING_3D_DATA_PATH = args.testing_3d_data_path
    TESTING_LABEL_PATH = args.testing_label_path
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    EPOCH_REPORT = args.epoch_report
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu') 

    logging.info(f'Model is currently using : {DEVICE}')

    paths = [SAVE_PATH]
    for path in paths:
        if not os.path.exists(path):
            logging.info(f'Creating Missing Paths: {path}')
            os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        logging.info(f'Saving config')
        json.dump(vars(args), file)
        file.close()
    
    # Prepare Training Data
    training_data = Human36MLoader(TRAINING_2D_DATA_PATH, TRAINING_3D_DATA_PATH, TRAINING_LABEL_PATH)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    # Prepare Testing Data
    testing_data = Human36MLoader(TESTING_2D_DATA_PATH, TESTING_3D_DATA_PATH, TESTING_LABEL_PATH)
    test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)
    
    logging.info(f'Setup Training and Testing Dataloaders')
    
    TOTAL_JOINTS = training_data.get_joint_numbers()
    TOTAL_ACTIONS = training_data.get_action_numbers()
    
    # Declare Model
    model = SimplePose(TOTAL_JOINTS, TOTAL_ACTIONS).to(DEVICE)
    # Apply Kaiming Init on Linear Layers
    model.apply(kaiming_weights_init)
    logging.info(f'Setup SimplePose model with Kaiming Weights')
    logging.info(model)
    
    # Declare Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    logging.info(f'Setup Optimizer')
    
    # Declare Scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96) # Value used by the original authors
    logging.info(f'Setup Scheduler')

    # Loss Function
    three_dim_pose_loss_fn = nn.MSELoss()
    action_label_loss_fn = nn.CrossEntropyLoss()
    logging.info(f'Setup loss functions')
    
    # Training and Testing Loop
    logging.info(f'Start Training and Testing Loops')
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_dict = {
            'model': model, 'dataloader': train_dataloader, 'device': DEVICE,  'optimizer': optimizer, 'three_dim_pose_loss_fn': three_dim_pose_loss_fn, 'action_label_loss_fn': action_label_loss_fn
        }
        train_predicted_labels, train_true_labels, train_total_losses, train_pose_losses, train_action_losses = train_once(train_dict)
        if epoch % EPOCH_REPORT == 0 or epoch == NUM_EPOCHS - 1:
            print_evaluation_metric(epoch, train_predicted_labels, train_true_labels, train_total_losses, train_pose_losses, train_action_losses, 'train')
            test_dict = {
                'model': model, 'dataloader': test_dataloader, 'device': DEVICE, 'three_dim_pose_loss_fn': three_dim_pose_loss_fn, 'action_label_loss_fn': action_label_loss_fn
            }
            test_predicted_labels, test_true_labels, test_total_losses, test_pose_losses, test_action_losses = test_once(test_dict)
            print_evaluation_metric(epoch, test_predicted_labels, test_true_labels, test_total_losses, test_pose_losses, test_action_losses, 'test')
            
            save_model(epoch, model, optimizer, scheduler)
        scheduler.step()

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    parser = argparse.ArgumentParser(description='SimplePose Training Code')
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--epoch_report', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_2d_data_path', type=str, default=os.path.join('datasets', 'h36m', 'Processed', 'train_2d_poses.npy'))
    parser.add_argument('--training_3d_data_path', type=str, default=os.path.join('datasets', 'h36m', 'Processed', 'train_3d_poses.npy'))
    parser.add_argument('--training_label_path', type=str, default=os.path.join('datasets', 'h36m', 'Processed', 'train_actions.npy'))
    parser.add_argument('--testing_2d_data_path', type=str, default=os.path.join('datasets', 'h36m', 'Processed', 'test_2d_poses.npy'))
    parser.add_argument('--testing_3d_data_path', type=str, default=os.path.join('datasets', 'h36m', 'Processed', 'test_3d_poses.npy'))
    parser.add_argument('--testing_label_path', type=str, default=os.path.join('datasets', 'h36m', 'Processed', 'test_actions.npy'))
    parser.add_argument('--save_path', type=str, default=os.path.join('model_outputs', 'simple_pose', timestamp))
    args = parser.parse_args()
    training_loop(args)
