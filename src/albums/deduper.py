import os
import argparse
import json
import numpy as np
from transformers import ViTFeatureExtractor, ViTModel
import torch
from sklearn.cluster import DBSCAN
from PIL import Image
import chainlit as cl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
# Check if MPS is available and set the device accordingly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize ViT model and feature extractor from Hugging Face
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)  # Move the model to MPS device

# JSON file to store image embeddings
EMBEDDINGS_JSON = "image_embeddings.json"

def load_embeddings():
    """Load existing embeddings from JSON file."""
    if os.path.exists(EMBEDDINGS_JSON):
        with open(EMBEDDINGS_JSON, 'r') as f:
            return json.load(f)
    return {}

def save_embeddings(embeddings_dict):
    """Save image embeddings to JSON file."""
    with open(EMBEDDINGS_JSON, 'w') as f:
        json.dump(embeddings_dict, f)

def extract_vit_features(image_paths):
    """Extract ViT features for a list of image paths."""
    images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
    inputs = feature_extractor(images=images, return_tensors="pt").to(device)  # Move input tensors to MPS device

    # Disable gradient tracking to save memory
    with torch.no_grad():
        features = vit_model(**inputs).last_hidden_state

    # Take the mean of the features to get a single vector representation
    feature_vectors = features.mean(dim=1).squeeze().cpu().numpy()  # Move the result back to CPU for compatibility with sklearn
    print(feature_vectors.shape)

    # Free up memory by deleting tensors that are no longer needed
    del inputs, features
    torch.cuda.empty_cache()  # Clear cache if running on GPU (useful in general, not MPS)

    return feature_vectors.reshape(1, -1)


def plot_k_distance(image_features, min_samples):
    # Fit a k-nearest neighbors model to the data
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(image_features)

    # Get the distances to the min_samples-th nearest neighbor
    distances, _ = neighbors.kneighbors(image_features)

    # Sort the distances and plot the k-distance plot
    distances = np.sort(distances[:, -1], axis=0)  # Take the distance to the k-th nearest neighbor
    plt.plot(distances)
    plt.title(f"K-distance plot (min_samples={min_samples})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("Distance to the k-th nearest neighbor")
    plt.show()

def cluster_images_dbscan(image_folder, batch_size=4, eps=2, min_samples=3):
    """Cluster images based on their ViT embeddings using DBSCAN."""
    # Load existing embeddings from JSON file
    embeddings_dict = load_embeddings()

    # Get the list of images in the folder (only .jpg images)
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    image_features = []
    image_paths_dict = {}

    # Process the images in batches and store their feature vectors
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        # Check for existing embeddings in JSON file for each image
        batch_embeddings = []
        for img_path in batch_paths:
            img_name = os.path.basename(img_path)
            if img_name in embeddings_dict:
                print(f"Reusing embedding for {img_name}")
                batch_embeddings.append(np.array(embeddings_dict[img_name]))  # Reuse existing embeddings
            else:
                # Extract features for the new image
                print(f"Extracting embedding for {img_name}")
                feature_vectors = extract_vit_features([img_path])
                batch_embeddings.append(feature_vectors[0])
                embeddings_dict[img_name] = feature_vectors.tolist()  # Store the embedding as list in JSON
        # Add feature vectors to the list
        batch_embeddings = np.array(batch_embeddings)
        image_features.extend(batch_embeddings.squeeze(axis=1))
        # image_features = np.array(image_features)
        
        # Store the corresponding image paths for reference
        for j, image_path in enumerate(batch_paths):
            image_paths_dict[len(image_features) - len(batch_paths) + j] = image_path

    # Save embeddings to JSON file after processing
    save_embeddings(embeddings_dict)

    # Perform DBSCAN clustering
    # Ensure that image_features is 2D before calling DBSCAN
    image_features = np.array(image_features)
    if image_features.ndim == 1:
        image_features = image_features.reshape(1, -1)  # If it's a single feature vector, make it 2D
    # print("shape",image_features.shape)
    # plot_k_distance(image_features, min_samples=3)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(image_features)

    # Group images by cluster (ignore noise points labeled as -1)
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label == -1:  # Ignore noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(image_paths_dict[idx])

    return clusters

# Define the necessary Chainlit callbacks
@cl.on_chat_start
async def on_chat_start():
    clusters = cluster_images_dbscan("./trans")
    print(clusters, "clusters")
    cl.user_session.set("clusters", clusters)
    cluster_ids =list(clusters.keys())
    cl.user_session.set("cluster_ids", cluster_ids )
    await cl.Message(content=f"Available clusters: {', '.join(map(str, cluster_ids))}").send()


@cl.on_message
async def on_message(message):
    selected_cluster = int(message.content.strip())
    cluster_ids = cl.user_session.get("cluster_ids")
    # Fetch the corresponding images for the selected cluster
    if selected_cluster in cluster_ids:
        selected_images = cl.user_session.get("clusters")[int(selected_cluster)]
        image_elements = []
        for img_path in selected_images:
            image_elements.append(cl.Image(path=img_path, name="user_image", display="inline"))
        await cl.Message(
            content=f"Displaying images from cluster {selected_cluster}:", elements=image_elements).send()

    else:
        await cl.Message(content="Invalid cluster ID. Please try again.").send()

def main():
    cl.chat()

if __name__ == "__main__":
    main()

