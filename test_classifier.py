import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import argparse


# ---------------------------- Dataset Preparation (as provided) ----------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, csv_path=None, good_root=None, not_good_root=None, transform=None):
        self.transform = transform
        self.data = []

        if csv_path:
            # Load the data using CSV
            self.labels_df = pd.read_csv(csv_path)
            self._load_images_from_csv(root_dir)
        else:
            # Load the data using directories
            self._load_images_from_dir(good_root, 0)
            self._load_images_from_dir(not_good_root, 1)

    def _load_images_from_csv(self, root_dir):
        for idx, row in self.labels_df.iterrows():
            img_name = os.path.join(root_dir, row['Image'])
            image = Image.open(img_name).convert('RGB')
            label = row['Label']
            self.data.append((image, label))

    def _load_images_from_dir(self, root, label):
        for fname in os.listdir(root):
            path = os.path.join(root, fname)
            image = Image.open(path).convert('RGB')
            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data(args):
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	testset = Dataset(root_dir=args.test_root, csv_path="test_label.csv", transform=transform)
	return testset


# ---------------------------- Model Preparation (as provided) ----------------------------
def prepare_model():
    resnet = torchvision.models.resnet101(pretrained=True)
    # for param in resnet.parameters():
    #     param.requires_grad = False

    num_features = resnet.fc.in_features
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 2)
    )
    return resnet

def load_trained_model(model_path):
	model = prepare_model()
	model.load_state_dict(torch.load(model_path))
	return model


# ---------------------------- Testing ----------------------------
def test_model(model, test_loader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	all_labels = []
	all_predictions = []

	with torch.no_grad():
		for inputs, labels in test_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			_, predicted = outputs.max(1)
			all_labels.extend(labels.cpu().numpy())
			all_predictions.extend(predicted.cpu().numpy())

	return all_labels, all_predictions


def main(args):
	test_data = prepare_data(args)
	test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

	model = load_trained_model('best_model.pth')
	labels, predictions = test_model(model, test_loader)

	# Compute metrics
	f1 = f1_score(labels, predictions)
	roc_auc = roc_auc_score(labels, predictions)
	print(f"F1 score: {f1:.4f}")
	print(f"ROC-AUC: {roc_auc:.4f}")

	# Save results
	results_df = pd.DataFrame({"True Labels": labels, "Predictions": predictions})
	results_df.to_csv("test_result.csv", index=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Test the screw classifier.")
	parser.add_argument("--test_root", default="archive-enhance/test", type=str, help="Path to the test folder.")
	parser.add_argument("--batch_size", default=100, type=int, help="Batch size for testing.")

	args = parser.parse_args()

	main(args)
