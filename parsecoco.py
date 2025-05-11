import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

class COCOParser:
    def __init__(self, annotation_file):
        """
        Initialize the COCO parser with the annotation file
        
        Args:
            annotation_file (str): Path to the COCO annotation JSON file
        """
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create dictionary mappings for easier access
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
    
    def get_image_info(self, image_id):
        """Get information about a specific image"""
        return self.images.get(image_id)
    
    def get_category_info(self, category_id):
        """Get information about a specific category"""
        return self.categories.get(category_id)
    
    def get_annotations(self, image_id):
        """Get all annotations for a specific image"""
        return self.annotations_by_image.get(image_id, [])
    
    def get_all_image_ids(self):
        """Get all image IDs in the dataset"""
        return list(self.images.keys())
    
    def get_all_category_ids(self):
        """Get all category IDs in the dataset"""
        return list(self.categories.keys())
    
    def visualize_image_annotations(self, image_id, image_dir, show_bbox=True, show_segmentation=True):
        """Visualize annotations for a specific image"""
        # Get image info and annotations
        image_info = self.get_image_info(image_id)
        if not image_info:
            print(f"Image ID {image_id} not found")
            return
        
        # Load the image
        image_path = os.path.join(image_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found")
            return
        
        img = Image.open(image_path)
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)
        
        # Get annotations for this image
        annotations = self.get_annotations(image_id)
        
        # Random colors for different categories
        colors = plt.cm.hsv(np.linspace(0, 1, len(self.categories) + 1))
        
        for ann in annotations:
            category_id = ann['category_id']
            category_name = self.categories[category_id]['name']
            color = colors[category_id % len(colors)]
            
            # Draw bounding box
            if show_bbox and 'bbox' in ann:
                bbox = ann['bbox']
                # COCO format: [x, y, width, height]
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(bbox[0], bbox[1] - 5, category_name, 
                        color='white', fontsize=10, 
                        bbox=dict(facecolor=color, alpha=0.7))
            
            # Draw segmentation
            if show_segmentation and 'segmentation' in ann:
                for segmentation in ann['segmentation']:
                    poly = np.array(segmentation).reshape(-1, 2)
                    ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
                    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.3)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def print_dataset_stats(self):
        """Print statistics about the dataset"""
        print("Dataset Statistics:")
        print(f"Number of images: {len(self.images)}")
        print(f"Number of categories: {len(self.categories)}")
        print(f"Number of annotations: {len(self.coco_data['annotations'])}")
        
        # Count instances per category
        instances_per_category = {}
        for ann in self.coco_data['annotations']:
            cat_id = ann['category_id']
            if cat_id not in instances_per_category:
                instances_per_category[cat_id] = 0
            instances_per_category[cat_id] += 1
        
        print("\nInstances per category:")
        for cat_id, count in instances_per_category.items():
            cat_name = self.categories[cat_id]['name']
            print(f"{cat_name}: {count}")


def main():
    """Example usage of the COCO parser"""
    # Replace these with your actual paths
    annotation_file = "test.json"
    image_dir = "test"
    
    parser = COCOParser(annotation_file)
    
    # Print dataset statistics
    parser.print_dataset_stats()
    
    # Get all image IDs
    image_ids = parser.get_all_image_ids()
    
    # If there are images, visualize the first one
    if image_ids:
        print(f"\nVisualizing annotations for image ID: {image_ids[0]}")
        parser.visualize_image_annotations(image_ids[0], image_dir)


if __name__ == "__main__":
    main()