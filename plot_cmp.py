import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot_compare(dataset_list,data_label,compare_type,dataset_name,bench_list):
    global L2_NN__TRAIN_ACC,L2_NN_TEST_ACC,L2_NN_SPARSITY,L2_NN_NEURONS
    global L1_NN__TRAIN_ACC,L1_NN_TEST_ACC,L1_NN_SPARSITY,L1_NN_NEURONS
    global SG_L1_NN__TRAIN_ACC,SG_L1_NN_TEST_ACC,SG_L1_NN_SPARSITY,SG_L1_NN_NEURONS
    
    if dataset_name=="sdd":
        L2_NN__TRAIN_ACC=0.98
        L2_NN_TEST_ACC=0.98
        L2_NN_SPARSITY=[0.17,0.36,0.36,0.16]
        L2_NN_NEURONS=[48.0,35.5,24.8,26.3]

        L1_NN__TRAIN_ACC=0.98
        L1_NN_TEST_ACC=0.98
        L1_NN_SPARSITY=[0.51,0.64,0.61,0.43]
        L1_NN_NEURONS=[47.9,27.5,19.6,20.2]

        SG_L1_NN__TRAIN_ACC=0.97
        SG_L1_NN_TEST_ACC=0.97
        SG_L1_NN_SPARSITY=[0.64,0.81,0.76,0.54]
        SG_L1_NN_NEURONS=[47.4,19.0,14.8,15.9]
    elif dataset_name=="mnist":
        L2_NN_TRAIN_ACC=0.99
        L2_NN_TEST_ACC=0.98
        L2_NN_SPARSITY=[0.60,0.60,0.34,0.08]
        L2_NN_NEURONS=[674.4,311,249.9,93.7]

        L1_NN__TRAIN_ACC=0.99
        L1_NN_TEST_ACC=0.97
        L1_NN_SPARSITY=[0.91,0.98,0.94,0.44]
        L1_NN_NEURONS=[658.2,84.8,85.1,73.3]

        SG_L1_NN__TRAIN_ACC=0.98
        SG_L1_NN_TEST_ACC=0.97
        SG_L1_NN_SPARSITY=[0.96,1.0,0.98,0.48]
        SG_L1_NN_NEURONS=[581.8,44.7,41.0,60.6]
    elif dataset_name=="covtype":
        L2_NN__TRAIN_ACC=0.85
        L2_NN_TEST_ACC=0.84
        L2_NN_SPARSITY=[0.04,0.10,0.22,0.14]
        L2_NN_NEURONS=[54.0,49.0,47.3,18.7]

        L1_NN__TRAIN_ACC=0.84
        L1_NN_TEST_ACC=0.83
        L1_NN_SPARSITY=[0.19,0.48,0.61,0.34]
        L1_NN_NEURONS=[53.0,46.0,31.0,14.3]

        SG_L1_NN__TRAIN_ACC=0.83
        SG_L1_NN_TEST_ACC=0.83
        SG_L1_NN_SPARSITY=[0.45,0.82,0.84,0.49]
        SG_L1_NN_NEURONS=[52.7,30.3,16.0,11.3]
    
    color_list=["deeppink","aqua","bisque"]
    marker_list=["*","v","o"]
    line_list=[">","<","-"]
    bench_label=[]
    show_label=[]
    
    fig=plt.figure(figsize=(8,6))
    for j,dataset in enumerate(dataset_list):
        color=color_list[j]
        marker=marker_list[j]
        if compare_type=="Feature_selection":
            acc=dataset["val_acc"]
            x=dataset["n1"]
            h1=plt.scatter(x,acc,c=color,marker=marker,label="1111")
            bench_label.append(h1)
            show_label.append(data_label[j])

        elif compare_type=="Hidden_layer_node_pruning":
            acc=dataset["val_acc"]
            x=dataset["n2"]+dataset["n3"]+dataset["n4"]
            h1=plt.scatter(x,acc,c=color,marker=marker)
            bench_label.append(h1)
            show_label.append(data_label[j])

        elif compare_type=="Sparsity":
            acc=dataset["val_acc"]
            x=dataset["sp1"]+dataset[:]["sp2"]+dataset["sp3"]+dataset["sp4"]
            h1=plt.scatter(x,acc,c=color,marker=marker)
            bench_label.append(h1)
            show_label.append(data_label[j])

    if "L1_NN" in bench_list:
        if compare_type=="Feature_selection":
            plt.axvline(L1_NN_NEURONS[0],c="red")
        elif compare_type=="Hidden_layer_node_pruning":
            plt.axvline(L1_NN_NEURONS[1]+L1_NN_NEURONS[2]+L1_NN_NEURONS[3],c="red")
        elif compare_type=="Sparsity":
            plt.axvline(L1_NN_SPARSITY[0]+L1_NN_SPARSITY[1]+L1_NN_SPARSITY[2]+L1_NN_SPARSITY[3],c="red")
        l1=plt.axhline(L1_NN_TEST_ACC,c="red")
        bench_label.append(l1)
        show_label.append("L1_NN")

    if "L2_NN" in bench_list:
        if compare_type=="Feature_selection":
            plt.axvline(L2_NN_NEURONS[0],c="green")
        elif compare_type=="Hidden_layer_node_pruning":
            plt.axvline(L2_NN_NEURONS[1]+L2_NN_NEURONS[2]+L2_NN_NEURONS[3],c="green")
        elif compare_type=="Sparsity":
            plt.axvline(L2_NN_SPARSITY[0]+L2_NN_SPARSITY[1]+L2_NN_SPARSITY[2]+L2_NN_SPARSITY[3],c="green")
        l2=plt.axhline(L2_NN_TEST_ACC,c="green")
        bench_label.append(l2)
        show_label.append("L2_NN")

    if "SG_L1" in bench_list:
        if compare_type=="Feature_selection":
            plt.axvline(SG_L1_NN_NEURONS[0],c="blue")
        elif compare_type=="Hidden_layer_node_pruning":
            plt.axvline(SG_L1_NN_NEURONS[1]+SG_L1_NN_NEURONS[2]+SG_L1_NN_NEURONS[3],c="blue")
        elif compare_type=="Sparsity":
            plt.axvline(SG_L1_NN_SPARSITY[0]+SG_L1_NN_SPARSITY[1]+SG_L1_NN_SPARSITY[2]+SG_L1_NN_SPARSITY[3],c="blue")
        l3=plt.axhline(SG_L1_NN_TEST_ACC,c="blue")
        bench_label.append(l3)
        show_label.append("SG_L1")

    plt.legend(handles=bench_label,labels=show_label,loc='best')
    plt.title(compare_type,fontsize=20)
    if compare_type=="Feature_selection":
        plt.xlabel("Total Number of fetures",fontsize=20)
    elif compare_type=="Hidden_layer_node_pruning":
        plt.xlabel("Total Number of Hidden node",fontsize=20)
    elif compare_type=="Sparsity":
        plt.xlabel("Total sparsity",fontsize=20)
    plt.ylabel("Accuracy",fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return fig