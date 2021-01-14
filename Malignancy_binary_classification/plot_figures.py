import matplotlib.pyplot as plt
import numpy as np

embedding_sampled = np.load("./log/pathology_binary/MASF/total_evaluation/embedding_sampled.npy")
labels1_sampled = np.load("./log/pathology_binary/MASF/total_evaluation/labels1_sampled.npy")
labels2_sampled = np.load("./log/pathology_binary/MASF/total_evaluation/labels2_sampled.npy")

class_list = {'0': 'benign', '1': 'malignant'}
domain_list = ['40', '100', '200', '400']
num_classes = 2

# _, ax = plt.subplots(1, figsize=(14, 10))
n_classes = num_classes
class_names = [class_list[str(i)] for i in range(len(class_list))]
# plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels1_sampled, cmap='bwr', alpha=1.0)
for class_index in range(n_classes):
    if class_index == 0:
        color_ = "r"
        marker_ = "^"
        size_ = 20
    else:
        color_ = "b"
        marker_ = "o"
        size_ = 20
    plt.scatter(embedding_sampled[labels1_sampled==class_index, 0], embedding_sampled[labels1_sampled==class_index, 1], s=size_, c=color_, marker=marker_, alpha=1.0)
plt.legend(class_names)
# cbar = plt.colorbar(boundaries=np.arange(num_classes + 1) - 0.5)
# cbar.set_ticks(np.arange(num_classes))
# cbar.set_ticklabels(class_names)
plt.xticks([])
plt.yticks([])
plt.show()

# _, ax = plt.subplots(1, figsize=(14, 10))
n_classes = len(domain_list)
class_names = domain_list
for class_index in range(n_classes):
    if class_index == 0:
        color_ = "r"
        marker_ = "^"
        size_ = 20
    elif class_index == 1:
        color_ = "b"
        marker_ = "o"
        size_ = 20
    elif class_index == 2:
        color_ = "g"
        marker_ = "s"
        size_ = 20
    else:
        color_ = "m"
        marker_ = "v"
        size_ = 20
    # plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels2_sampled, cmap='Spectral', alpha=1.0)
    plt.scatter(embedding_sampled[labels2_sampled==class_index, 0], embedding_sampled[labels2_sampled==class_index, 1], s=size_, c=color_, marker=marker_, alpha=1.0)
plt.legend(class_names)
# cbar = plt.colorbar(boundaries=np.arange(n_classes + 1) - 0.5)
# cbar.set_ticks(np.arange(n_classes))
# cbar.set_ticklabels(class_names)
plt.xticks([])
plt.yticks([])
plt.show()