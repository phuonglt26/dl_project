import numpy as np
from sklearn.model_selection import train_test_split

NUM_IMAGE = 5
def get_num_samples_from_prior_phase(features, labels, num_image):
    unique_classes = np.unique(labels)

    # Use a list comprehension to select 5 samples for each class
    selected_samples = [
        np.where(labels == class_label)[0][:num_image] for class_label in unique_classes
    ]

    # Concatenate the selected indices
    selected_samples = np.concatenate(selected_samples)

    # Now, 'selected_samples' contains the indices of 5 samples from each class

    # Access the corresponding feature vectors and labels
    selected_features = features[selected_samples]
    selected_labels = labels[selected_samples]
    return selected_features, selected_labels


def shuffle_data(features, labels):
    # Create an index array and shuffle it
    indices = np.arange(len(features))
    np.random.shuffle(indices)

    # Use the shuffled indices to rearrange your feature and label arrays
    shuffled_features = features[indices]
    shuffled_labels = labels[indices]
    return shuffled_features, shuffled_labels


def train_val_split():
    data_dir = "data/preprocessed_train/"
    save_train_val_phase_dir = "data/train_val_phase/"
    save_train_phase_dir = "data/train_phase/"
    buffer_memory_data = None
    buffer_memory_labels = None
    for phase_num in range(1, 11):
        print(f"Starting train val split phase {phase_num} ...")
        X = np.load(data_dir + f"phase_{phase_num}.npy")
        y = np.load(data_dir + f"label_phase_{phase_num}.npy")

        X, y = shuffle_data(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        if phase_num > 1:
            if buffer_memory_data is None:
                buffer_memory_data = prior_data
                buffer_memory_labels = prior_labels
            else:
                buffer_memory_data = np.concatenate(
                    (buffer_memory_data, prior_data), axis=0
                )
                buffer_memory_labels = np.concatenate(
                    (buffer_memory_labels, prior_labels), axis=0
                )
            X = np.concatenate((X, buffer_memory_data), axis=0)
            y = np.concatenate((y, buffer_memory_labels), axis=0)

            X_train = np.concatenate((X_train, buffer_memory_data), axis=0)
            y_train = np.concatenate((y_train, buffer_memory_labels), axis=0)

            # get all previous validation to test this phase
            X_test = np.concatenate((X_test, prior_val), axis=0)
            y_test = np.concatenate((y_test, prior_val_labels), axis=0)

        np.save(save_train_phase_dir + f"phase_{phase_num}.npy", X)
        np.save(save_train_phase_dir + f"label_phase_{phase_num}.npy", y)

        np.save(save_train_val_phase_dir + f"train_phase_{phase_num}.npy", X_train)
        np.save(save_train_val_phase_dir + f"val_phase_{phase_num}.npy", X_test)
        np.save(
            save_train_val_phase_dir + f"train_label_phase_{phase_num}.npy", y_train
        )
        np.save(save_train_val_phase_dir + f"val_label_phase_{phase_num}.npy", y_test)

        prior_data, prior_labels = get_num_samples_from_prior_phase(
            X, y, num_image=NUM_IMAGE
        )
        prior_val = X_test
        prior_val_labels = y_test

        print(f"Saved train test splitting phase {phase_num}.")


train_val_split()
