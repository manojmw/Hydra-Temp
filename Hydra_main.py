#!/usr/bin/python

##############################################

# Manoj M Wagle (USydney; CMRI)

##############################################


import os, sys, random, time
import argparse
import logging
import glob
from captum.attr import *
import pandas as pd
import numpy as np
import h5py


##############################################

# Create argument parser
parser = argparse.ArgumentParser("Hydra")

parser.add_argument('--seed', type = int, default = 42, help ='seed')

# Input Data
parser.add_argument('--base-dir', metavar = 'DIR', default = '/media/disk4/manojw/RareCT_Classification/Original_Data/PBMC-TEASeq_Swanson/batch_1-2', help = 'path to base directory') 

# Training
parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
parser.add_argument('--epochs', type = int, default = 40, help = 'num of training epochs')
parser.add_argument('--lr', type = float, default = 0.02, help = 'learning rate')

# GPU specification    
parser.add_argument('--gpus', type = str, default = '1', help = 'Comma-separated list of GPU indices to be used (Ex: "--gpus 0,1,2" to use the first 3 GPUs ). If not specified, all GPUs will be used')    

# Model
parser.add_argument('--z_dim', type = int, default = 100, help = 'the number of neurons in latent space')
parser.add_argument('--hidden_rna', type = int, default = 185, help = 'the number of neurons for RNA layer')
parser.add_argument('--hidden_adt', type = int, default = 30, help = 'the number of neurons for ADT layer')
parser.add_argument('--hidden_atac', type = int, default = 185, help = 'the number of neurons for ATAC layer')
parser.add_argument('--num_models', type = int, default = 25, help= 'number of models for Ensemble Learning')

# Task 
parser.add_argument('--setting', type=str, default= "train", help='Whether training or testing')

# Parse the command-line arguments
args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


##############################################

# Import torch-related libraries after setting the CUDA_VISIBLE_DEVICES
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from learn.model import (Autoencoder_CITEseq_Step1, Autoencoder_SHAREseq_Step1, Autoencoder_TEAseq_Step1,
                         Autoencoder_CITEseq_Step2, Autoencoder_SHAREseq_Step2, Autoencoder_TEAseq_Step2, 
                         Autoencoder_RNAseq_Step1, Autoencoder_RNAseq_Step2, Autoencoder_ADTseq_Step1,
                         Autoencoder_ADTseq_Step2, Autoencoder_ATACseq_Step1, Autoencoder_ATACseq_Step2)
from learn.train import train_model
from util import (MyDataset, read_h5_data, Index2Label, read_fs_label,
                  load_and_preprocess_data, perform_data_augmentation, setup_seed)

# Check the device type based on GPU availability, MPS availability, or defaulting to CPU
device_str = "CUDA" if torch.cuda.is_available() \
            else "MPS" if getattr(torch, 'has_mps', False)() \
            else "CPU"
device = torch.device(device_str.lower())

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

setup_seed(args.seed) ### set random seed for reproducbility


##############################################

specified_gpus = args.gpus.split(',') if args.gpus else []
num_gpus_specified = len(specified_gpus)

# If CUDA_VISIBLE_DEVICES is not set, use PyTorch to get the total GPU count.
if not specified_gpus:
    num_gpus = torch.cuda.device_count()
else:
    num_gpus = num_gpus_specified

# Get the indices of GPUs that are currently visible to PyTorch
active_gpu_indices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')

print("=====================================")
print("\nNumber of GPUs available:", num_gpus)

if num_gpus >= 1 and (device_str == "CPU" or device_str == "MPS"):
    print("It seems the CPU version of PyTorch is installed. For GPU utilization, \
        please install the GPU version of PyTorch. Currently, running on CPU!!!")

if num_gpus > 1:
    print("GPU Parallel Computing: YES")
else: 
    print("GPU Parallel Computing: NO")

print("Device to be used:", device_str, "\n")
print("=====================================\n")


##############################################

def main(args):

    logging.info("Starting to run...")

    # Get a list of all dataset folders
    dataset_folders = sorted(glob.glob(f"{args.base_dir}/dataset_1"))

    for dataset_folder in dataset_folders:

        logging.info("Processing %s" % dataset_folder)

        dataset_cell_results = []
        dataset_cell_type_results = []
        dataset_feature_selection_results = {}

        # Get a list of all split folders within the current dataset folder
        split_folders = sorted(glob.glob(f"{dataset_folder}/split_*"))

        training_times = []

        for split_folder in split_folders:

            # if split_folder == "/dskh/nobackup/manojw/RareCT_Classification/Original_Data/FetalLungCellAtlas_He/dataset_1/split_4" or split_folder == "/dskh/nobackup/manojw/RareCT_Classification/Original_Data/FetalLungCellAtlas_He/dataset_1/split_5":

            logging.info("Process %s" % split_folder)

            model_save_path = f"{split_folder}/trained_model/"
            os.makedirs(model_save_path, exist_ok = True)

            if args.setting == 'train':

                if os.path.isfile(f"{split_folder}/rna_train.h5"):
                    args.rna = f"{split_folder}/rna_train.h5"
                else:
                    args.rna = "NULL" 
                if os.path.isfile(f"{split_folder}/adt_train.h5"):
                    args.adt = f"{split_folder}/adt_train.h5"
                else:
                    args.adt = "NULL" 
                if os.path.isfile(f"{split_folder}/atac_train.h5"):
                    args.atac = f"{split_folder}/atac_train.h5"
                else:
                    args.atac = "NULL"
                args.cty = f"{split_folder}/ct_train.csv"


                if args.adt != "NULL" and args.atac != "NULL": # TEAseq
                    # Load and preprocess the data
                    (train_data, train_dl, train_label, mode, classify_dim, nfeatures_rna, nfeatures_adt, nfeatures_atac, feature_num, label_to_name_mapping) = load_and_preprocess_data(args, setting = "train")
                    logging.info("The Dataset is: %s" % mode)

                if args.adt != "NULL" and args.atac == "NULL": # CITEseq
                    # Load and preprocess the data
                    (train_data, train_dl, train_label, mode, classify_dim, nfeatures_rna, nfeatures_adt, feature_num, label_to_name_mapping) = load_and_preprocess_data(args, setting = "train")
                    logging.info("The Dataset is: %s" % mode)

                if args.adt == "NULL" and args.atac != "NULL": # SHAREseq
                    # Load and preprocess the data
                    (train_data, train_dl, train_label, mode, classify_dim, nfeatures_rna, nfeatures_atac, feature_num, label_to_name_mapping) = load_and_preprocess_data(args, setting = "train")
                    logging.info("The Dataset is: %s" % mode)
                                
                if args.adt == "NULL" and args.atac == "NULL": # scRNA-seq
                    # Load and preprocess the data
                    (train_data, train_dl, train_label, mode, classify_dim, nfeatures_rna, feature_num, label_to_name_mapping) = load_and_preprocess_data(args, setting = "train")
                    logging.info("The Dataset is: %s" % mode)

                if args.atac == "NULL" and args.rna == "NULL": # scADT-Seq
                    # Load and preprocess the data
                    (train_data, train_dl, train_label, mode, classify_dim, nfeatures_adt, feature_num, label_to_name_mapping) = load_and_preprocess_data(args, setting = "train")
                    logging.info("The Dataset is: %s" % mode)

                if args.adt == "NULL" and args.rna == "NULL": # scATAC-Seq
                    # Load and preprocess the data
                    (train_data, train_dl, train_label, mode, classify_dim, nfeatures_atac, feature_num, label_to_name_mapping) = load_and_preprocess_data(args, setting = "train")
                    logging.info("The Dataset is: %s" % mode)


                start_time = time.time()  # start time measurement
                
                                
                ########## Step 1 ########### 

                ### Build model
                if mode == "TEAseq":
                    model = Autoencoder_TEAseq_Step1(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim, args.num_models)
                elif mode == "CITEseq":
                    model = Autoencoder_CITEseq_Step1(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim, args.num_models)
                elif mode == "SHAREseq":
                    model = Autoencoder_SHAREseq_Step1(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim, args.num_models)
                elif mode == "scRNA-Seq":
                    model = Autoencoder_RNAseq_Step1(nfeatures_rna, args.hidden_rna, args.z_dim, classify_dim, args.num_models)
                elif mode == "ADT-Seq":
                    model = Autoencoder_ADTseq_Step1(nfeatures_adt, args.hidden_adt, args.z_dim, classify_dim, args.num_models)
                elif mode == "ATAC-Seq":
                    model = Autoencoder_ATACseq_Step1(nfeatures_atac, args.hidden_atac, args.z_dim, classify_dim, args.num_models)


                # Move the model to GPU and automatically utilize multiple GPUs if available
                if torch.cuda.device_count() > 1:
                    model = model.to(device)
                    model = nn.DataParallel(model)
                else:
                    model = model.to(device)

                logging.info("Now training on raw data...")

                # Train the original model on the original raw data including all classifiers
                model, loss, train_num = train_model(model, train_dl, lr=args.lr, epochs=args.epochs, 
                        classify_dim=classify_dim, save_path=model_save_path, 
                        save_filename='Original_Model.pth.tar', feature_num=feature_num, 
                        use_curriculum=True)

            
                checkpoint_tar = os.path.join(model_save_path, 'Original_Model.pth.tar')
                if os.path.exists(checkpoint_tar):
                    # Load the model's weights
                    checkpoint = torch.load(checkpoint_tar)
                    model = model.module if isinstance(model, nn.DataParallel) else model
                    model.load_state_dict(checkpoint['state_dict'], strict=True)


                ########## Step 2 ###########

                # define the range of epochs
                min_epochs, max_epochs = 30, 50

                for modelI in range(args.num_models):

                    logging.info("Refining Model: %s...", modelI+1)
                    
                    if mode == "TEAseq":
                        Step2_model = Autoencoder_TEAseq_Step2(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim)
                    elif mode == "CITEseq":
                        Step2_model = Autoencoder_CITEseq_Step2(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim)
                    elif mode == "SHAREseq":
                        Step2_model = Autoencoder_SHAREseq_Step2(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim)
                    elif mode == "scRNA-Seq":
                        Step2_model = Autoencoder_RNAseq_Step2(nfeatures_rna, args.hidden_rna, args.z_dim, classify_dim)
                    elif mode == "scADT-Seq":
                        Step2_model = Autoencoder_ADTseq_Step2(nfeatures_adt, args.hidden_adt, args.z_dim, classify_dim)
                    elif mode == "scATAC-Seq":
                        Step2_model = Autoencoder_ATACseq_Step2(nfeatures_atac, args.hidden_atac, args.z_dim, classify_dim)

                    # Load the encoder weights trained from Step 1 
                    encoder_weights = model.encoder.state_dict() 
                    Step2_model.encoder.load_state_dict(encoder_weights)

                    # Load the decoder weights trained from Step 1 
                    decoder_weights = model.decoder.state_dict() 
                    Step2_model.decoder.load_state_dict(decoder_weights)

                    # Load the classifier weights trained from Step 1 
                    classifier_weights = model.classifiers[modelI].state_dict()
                    Step2_model.classify.load_state_dict(classifier_weights)

                    # Move the model to GPU and automatically utilize multiple GPUs if available
                    if torch.cuda.device_count() > 1:
                        Step2_model = Step2_model.to(device)
                        Step2_model = nn.DataParallel(Step2_model)
                    else:    
                        Step2_model = Step2_model.to(device)
                                                        
                    # Generate Augmented dataset  
                    new_data, new_label, new_label_names = perform_data_augmentation(label_to_name_mapping, train_num, classify_dim, train_label, train_data, model, args)

                    # Process the new data after augmentation
                    train_transformed_dataset = MyDataset(new_data, new_label)

                    new_train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last = False)

                    # set a random number of epochs for this model
                    epochs = random.randint(min_epochs, max_epochs)

                    Step2_model, loss, _ = train_model(Step2_model, new_train_dl, lr=args.lr, epochs=epochs, 
                                            classify_dim=classify_dim, save_path=model_save_path, 
                                            save_filename='Curriculum_Model.pth.tar', feature_num=feature_num, 
                                            use_curriculum=False)
                    

                    # Create directory for balanced data if it does not exist
                    balanced_data_dir = os.path.join(model_save_path, 'Balanced_Data-25')
                    os.makedirs(balanced_data_dir, exist_ok = True)
                    
                    new_data, new_label, new_label_names = perform_data_augmentation(label_to_name_mapping, train_num, classify_dim, train_label, train_data, Step2_model, args)

                    # # Convert new_label to numpy array if it's not already
                    # new_label_np = new_label.cpu().numpy()

                    # # Compute unique cell types and their counts
                    # unique_labels, counts = np.unique(new_label_np, return_counts=True)

                    # # Print the cell type and count
                    # for label, count in zip(unique_labels, counts):
                    #     print(f"Cell Type: {label}, Count: {count}")

                    # Modify new_data_path and new_label_path to use balanced_data_dir
                    new_data_path = os.path.join(balanced_data_dir, f'train_data_bal_{modelI+1}.pt')
                    torch.save(new_data, new_data_path)

                    new_label_path = os.path.join(balanced_data_dir, f'train_label_bal_{modelI+1}.pt')
                    torch.save(new_label, new_label_path)

                    # # Create a DataFrame from the new labels and new label names
                    # new_label_df = pd.DataFrame({
                    #     '': range(1, len(new_label_names) + 1),  # Mimic the same structure as original label CSV
                    #     'x': new_label_names
                    # })
                    # #  Save the DataFrame to a CSV file
                    # new_label_path = os.path.join(balanced_data_dir, f'train_label_bal_{modelI+1}.csv')
                    # new_label_df.to_csv(new_label_path, index=False)                                            
                
                    # Create directory for final models if it does not exist
                    final_models_dir = os.path.join(model_save_path, 'Final_Models-25')
                    os.makedirs(final_models_dir, exist_ok = True)

                    final_model = Step2_model.module if isinstance(Step2_model, nn.DataParallel) else Step2_model

                    # Save the final model
                    final_model_path = os.path.join(final_models_dir, f'model_{modelI+1}.pth.tar')
                    torch.save({'state_dict': final_model.state_dict()}, final_model_path)

                end_time = time.time()  # end time measurement
                training_time = end_time - start_time
                
                logging.info("Training time: %s seconds" % training_time)

                training_times.append(training_time)


                ############ Perform feature selection ############  

                logging.info("Running feature selection...")

                # Convert feature names to string format
                rna_name_new = []
                adt_name_new = []
                atac_name_new = []

                if os.path.isfile(f"{split_folder}/rna_train.h5"):
                    rna_data_path = f"{split_folder}/rna_train.h5"
                    rna_data_path_noscale = f"{split_folder}/rna_train_noscale.h5"
                else:
                    rna_data_path = "NULL" 
                if os.path.isfile(f"{split_folder}/adt_train.h5"):
                    adt_data_path = f"{split_folder}/adt_train.h5"
                    adt_data_path_noscale = f"{split_folder}/adt_train_noscale.h5"
                else:
                    adt_data_path = "NULL" 
                if os.path.isfile(f"{split_folder}/atac_train.h5"):
                    atac_data_path = f"{split_folder}/atac_train.h5"
                    atac_data_path_noscale = f"{split_folder}/atac_train_noscale.h5"
                else:
                    atac_data_path = "NULL"
                label_path = f"{split_folder}/ct_train.csv"

                (label, _) = read_fs_label(label_path)

                index_to_label = Index2Label(label_path, classify_dim)

                classify_dim = (max(label)+1).cpu().numpy()

                if adt_data_path != "NULL" and atac_data_path != "NULL":
                    mode = "TEAseq"

                    rna_name = h5py.File(rna_data_path, "r")['matrix/features'][:]
                    adt_name = h5py.File(adt_data_path, "r")['matrix/features'][:]
                    atac_name = h5py.File(atac_data_path, "r")['matrix/features'][:]

                    rna_data = read_h5_data(rna_data_path)
                    adt_data = read_h5_data(adt_data_path)
                    atac_data = read_h5_data(atac_data_path)

                    rna_data_noscale = read_h5_data(rna_data_path_noscale)
                    adt_data_noscale = read_h5_data(adt_data_path_noscale)
                    atac_data_noscale = read_h5_data(atac_data_path_noscale)

                    nfeatures_rna = rna_data.shape[1]
                    nfeatures_adt = adt_data.shape[1]
                    nfeatures_atac = atac_data.shape[1]

                    feature_num = nfeatures_rna + nfeatures_adt + nfeatures_atac

                    data = torch.cat((rna_data, adt_data, atac_data), 1)
                    data_noscale = torch.cat((rna_data_noscale, adt_data_noscale, atac_data_noscale), 1)
                
                if adt_data_path != "NULL" and atac_data_path == "NULL":
                    mode = "CITEseq"

                    rna_name = h5py.File(rna_data_path, "r")['matrix/features'][:]
                    adt_name = h5py.File(adt_data_path, "r")['matrix/features'][:]

                    rna_data = read_h5_data(rna_data_path)
                    adt_data = read_h5_data(adt_data_path)

                    rna_data_noscale = read_h5_data(rna_data_path_noscale)
                    adt_data_noscale = read_h5_data(adt_data_path_noscale)

                    nfeatures_rna = rna_data.shape[1]
                    nfeatures_adt = adt_data.shape[1]

                    feature_num = nfeatures_rna + nfeatures_adt

                    data = torch.cat((rna_data, adt_data), 1)
                    data_noscale = torch.cat((rna_data_noscale, adt_data_noscale), 1)
                
                if adt_data_path == "NULL" and atac_data_path != "NULL":
                    mode = "SHAREseq"

                    rna_name = h5py.File(rna_data_path, "r")['matrix/features'][:]
                    atac_name = h5py.File(atac_data_path, "r")['matrix/features'][:]

                    rna_data = read_h5_data(rna_data_path)
                    atac_data = read_h5_data(atac_data_path)

                    rna_data_noscale = read_h5_data(rna_data_path_noscale)
                    atac_data_noscale = read_h5_data(atac_data_path_noscale)

                    nfeatures_rna = rna_data.shape[1]
                    nfeatures_atac = atac_data.shape[1]

                    feature_num = nfeatures_rna + nfeatures_atac

                    data = torch.cat((rna_data, atac_data), 1)
                    data_noscale = torch.cat((rna_data_noscale, atac_data_noscale), 1)
                
                if adt_data_path == "NULL" and atac_data_path == "NULL":
                    mode = "scRNA-Seq"

                    rna_name = h5py.File(rna_data_path, "r")['matrix/features'][:]

                    rna_data = read_h5_data(rna_data_path)
                    rna_data_noscale = read_h5_data(rna_data_path_noscale)

                    nfeatures_rna = rna_data.shape[1]

                    feature_num = nfeatures_rna

                    data = rna_data
                    data_noscale = rna_data_noscale


                for i in range(nfeatures_rna):
                    a = str(rna_name[i], encoding="utf-8") + "_RNA_"
                    rna_name_new.append(a)

                if mode == "CITEseq":
                    for i in range(nfeatures_adt):
                        a = str(adt_name[i], encoding="utf-8") + "_ADT_"
                        adt_name_new.append(a)
                    features = rna_name_new + adt_name_new

                if mode == "SHAREseq":
                    for i in range(nfeatures_atac):
                        a = str(atac_name[i], encoding="utf-8") + "_ATAC_"
                        atac_name_new.append(a)
                    features = rna_name_new + atac_name_new

                if mode == "TEAseq":
                    for i in range(nfeatures_adt):
                        a = str(adt_name[i], encoding="utf-8") + "_ADT_"
                        adt_name_new.append(a)
                    for i in range(nfeatures_atac):
                        a = str(atac_name[i], encoding="utf-8") + "_ATAC_"
                        atac_name_new.append(a)
                    features = rna_name_new + adt_name_new + atac_name_new
                
                if mode == "scRNA-Seq":
                    features = rna_name_new

                # Load all model files
                model_files = glob.glob(os.path.join(model_save_path, 'Final_Models-25/*.pth.tar'))

                # Perform feature selection for each cell type
                for i in range(classify_dim):

                    cell_type_name = index_to_label[i]

                    # Select the data for the current cell type and for all other cell types
                    current_type_data = data_noscale[torch.where(label == i)].reshape(-1, feature_num)
                    other_type_data = data_noscale[torch.where(label != i)].reshape(-1, feature_num)

                    # Calculate the mean expression for each feature across the two groups
                    mean_current_type = torch.mean(current_type_data, dim=0)
                    mean_other_types = torch.mean(other_type_data, dim=0)

                    # Compute fold changes for each feature
                    epsilon = 1e-6
                    fold_changes = (mean_current_type + epsilon) / (mean_other_types + epsilon)

                    # Apply log transformation to fold changes
                    log_fold_changes = torch.log2(fold_changes)

                    # Get indices of cells with the current cell type
                    train_index_fs = torch.where(label == i)
                    train_index_fs = [t.cpu().numpy() for t in train_index_fs]
                    train_index_fs = np.array(train_index_fs)

                    # Get data for the current cell type
                    train_data_each_celltype_fs = data[train_index_fs, :].reshape(-1, feature_num)

                    attributions_all_models = []

                    # Compute the attribution for each cell of the current cell type
                    for model_file in model_files:
                        if mode == "CITEseq":
                            model_test = Autoencoder_CITEseq_Step2(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim)
                        elif mode == "SHAREseq":
                            model_test = Autoencoder_SHAREseq_Step2(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim)
                        elif mode == "TEAseq":
                            model_test = Autoencoder_TEAseq_Step2(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim)
                        elif mode == "scRNA-Seq":
                            model_test = Autoencoder_RNAseq_Step2(nfeatures_rna, args.hidden_rna, args.z_dim, classify_dim)
                        elif mode == "scADT-Seq":
                            model_test = Autoencoder_ADTseq_Step2(nfeatures_adt, args.hidden_adt, args.z_dim, classify_dim)
                        elif mode == "scATAC-Seq":
                            model_test = Autoencoder_ATACseq_Step2(nfeatures_atac, args.hidden_atac, args.z_dim, classify_dim)

                        # Load the model's weights
                        checkpoint = torch.load(model_file)
                        model_test.load_state_dict(checkpoint['state_dict'], strict=True)

                        model_test = model_test.to(device)

                        classify_model = nn.Sequential(*list(model_test.children()))[0:2]

                        deconv = IntegratedGradients(classify_model)

                        batch_size = 500  # adjust this based on your GPU memory

                        # Initialize the attributions tensor
                        attribution = torch.zeros(1, feature_num).cuda()

                        # Calculate the attributions in batches
                        for j in range(0, train_data_each_celltype_fs.size(0), batch_size):
                            batch = train_data_each_celltype_fs[j:j + batch_size, :]
                            batch = batch.to(device)
                            attribution += torch.sum(torch.abs(deconv.attribute(batch, target=i)), dim=0, keepdim=True)

                        # take mean attribution for the current model
                        attribution_mean = torch.mean(attribution, dim=0)
                        attributions_all_models.append(attribution_mean)

                        del model_test  # delete the current model to free up memory
                        torch.cuda.empty_cache()  # empty GPU cache to avoid out-of-memory errors

                    # calculate the average attribution across all models
                    average_attribution = sum(attributions_all_models) / len(model_files)

                    fs_score = average_attribution.reshape(-1).detach().cpu().numpy()

                    # Adjust by the sign of the log fold changes
                    fs_score = fs_score * np.sign(log_fold_changes.numpy())

                    # Directly sort by the scores themselves
                    indices_sorted = np.argsort(-fs_score)  # This sorts the scores in descending order

                    # Use the sorted indices to order your features and scores
                    fs_results = [features[index] + str(index) for index in indices_sorted]
                    fs_scores = [fs_score[index] for index in indices_sorted]

                    # Convert fs_results to a pandas DataFrame
                    fs_results_df = pd.DataFrame({'Feature Name': fs_results, 'Score': fs_scores})

                    # Extract the base name of the split folder
                    split_folder_name = os.path.basename(split_folder)

                    # Create "Hydra" directory inside the dataset folder
                    aurora_folder = os.path.join(dataset_folder, "Hydra_Ensemble-25")

                    save_fs_eachcell = os.path.join(aurora_folder, split_folder_name)
                    if not os.path.exists(save_fs_eachcell):
                        os.makedirs(save_fs_eachcell)

                    # Save the feature selection results to a CSV file
                    fs_results_df.to_csv(os.path.join(save_fs_eachcell, f'fs.{cell_type_name}_Hydra_new.csv'), index=False)

                    # Append the results to the dataset feature selection results
                    if cell_type_name in dataset_feature_selection_results:
                        dataset_feature_selection_results[cell_type_name].append(fs_results_df)
                    else:
                        dataset_feature_selection_results[cell_type_name] = [fs_results_df]


        with open(f"{dataset_folder}/training_times.txt", 'w') as f:
            for training_time in training_times:
                f.write(str(training_time) + '\n')

    logging.info("Completed successfully!")

    return


##############################################

if __name__ == "__main__":
    # Logging to Standard Error
    Log_Format = "%(levelname)s - %(asctime)s - %(message)s \n"
    logging.basicConfig(stream = sys.stderr, format = Log_Format, level = logging.INFO)

    # Call the main function
    main(args)
