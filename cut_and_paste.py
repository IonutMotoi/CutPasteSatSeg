import numpy as np
import os
import glob


class CutAndPaste:
    def __init__(self, cap_config):
        self.root = cap_config["root"]
        self.classes = cap_config["classes"]
        self.num_of_instances = cap_config["num_of_instances"]
        self.augment_instances = cap_config["augment_instances"]

        self.class_instances = {}
        for class_index in self.classes:
            self.class_instances[class_index] = glob.glob(os.path.join(
                self.root, f"class_{str(class_index)}", "*[!mask].npy"))

    def __len__(self):
        # Returns the total number of instances across all classes
        return sum(len(instances) for instances in self.class_instances.values())

    def paste_instances(self, dest_img, dest_label):
        for i in range(self.num_of_instances):
            # Sample a random instance
            class_index = np.random.choice(self.classes)
            instance_index = np.random.choice(len(self.class_instances[class_index]))

            # Load the instance
            img_path = self.class_instances[class_index][instance_index]
            mask_path = img_path.replace(".npy", "_mask.npy")
            instance = np.load(img_path)
            instance_mask = np.load(mask_path)

            # Augment the instance
            if self.augment_instances:
                instance, instance_mask = self._augment_instance(instance, instance_mask)

            # Sample a random position where to paste the instance
            top = np.random.randint(0, dest_img.shape[1] - instance.shape[1] + 1)
            left = np.random.randint(0, dest_img.shape[2] - instance.shape[2] + 1)

            # Paste the instance
            slice_idx = np.index_exp[:, top:top+instance.shape[1], left:left+instance.shape[2]]
            dest_img[slice_idx] = np.where(instance_mask, instance, dest_img[slice_idx])
            dest_label[slice_idx[1:]] = np.where(instance_mask, class_index, dest_label[slice_idx[1:]])

    def _augment_instance(self, image, mask):
        image = image.transpose(1, 2, 0)

        # Horizontal Flip
        if np.random.rand() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # # Vertical Flip
        if np.random.rand() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # # Random Rotate 90
        k = np.random.randint(4)  # Randomly choose between 0, 1, 2, or 3
        image = np.rot90(image, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

        image = image.transpose(2, 0, 1)
        return image, mask
