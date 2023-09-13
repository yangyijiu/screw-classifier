from imports import *
from logger_setup import *
from data_utils import *
from evaluation import *


# ---------------------------- Dataset Preparation ----------------------------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, good_root, not_good_root, transform=None):
        self.transform = transform
        self.data = []
        self._load_images(good_root, 0)
        self._load_images(not_good_root, 1)

    def _load_images(self, root, label):
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
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SimpleDataset(args.good_root, args.not_good_root, transform=transform)
    testset = SimpleDataset(args.test_good_root, args.test_not_good_root, transform=transform)

    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, val_size])
    return train_data, val_data, testset


# ---------------------------- Model Preparation ----------------------------
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


# ---------------------------- Training & Evaluation ----------------------------
def train_model(model, train_loader, val_loader,test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weights = torch.tensor([0.2, 0.8])  # Assuming class 0 is the majority class
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    best_auc_roc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log_metrics(epoch, total_loss, train_loader, model, val_loader, test_loader, writer)

        current_val_auc_roc = compute_auc_roc(model, val_loader)
        if current_val_auc_roc > best_auc_roc:
            best_auc_roc = current_val_auc_roc
            print(f"Saving the best model with auc_roc: {best_auc_roc:.4f}")
            torch.save(model.state_dict(), 'best_model.pth')

# ---------------------------- Main Function ----------------------------
def main(args):
    train_data, val_data, test_data = prepare_data(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    model = prepare_model()
    train_model(model, train_loader, val_loader,test_loader, args)

    model.load_state_dict(torch.load('best_model.pth'))
    # ======================= Validation =======================
    f1_val = evaluate(model, val_loader)
    val_auc_roc = compute_auc_roc(model, val_loader)
    print(f"Validation F1 score: {f1_val:.4f}")
    print(f"Validation AUC-ROC: {val_auc_roc:.4f}")
    logging.info(f'Validation F1 score: {f1_val:.4f},Validation AUC-ROC: {val_auc_roc:.4f}')


    # ======================= Testing =======================
    f1_test = evaluate(model, test_loader)
    test_auc_roc = compute_auc_roc(model, test_loader)
    print(f"Test F1 score: {f1_test:.4f}")
    print(f"Test AUC-ROC: {test_auc_roc:.4f}")
    logging.info(f'Test F1 score: {f1_test:.4f},Test AUC-ROC: {test_auc_roc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a screw classifier.")
    parser.add_argument("--good_root", default="./archive-diff/train/good", type=str, help="Path to the good screws folder.")
    parser.add_argument("--not_good_root", default="./archive-diff/train/not-good", type=str,help="Path to the not-good screws folder.")
    parser.add_argument("--test_good_root", default="./archive-diff/test/good", type=str, help="Path to the good screws folder.")
    parser.add_argument("--test_not_good_root", default="./archive-diff/test/not-good", type=str,help="Path to the not-good screws folder.")
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate for the optimizer.")
    parser.add_argument("--train_split", default=0.8, type=float, help="Proportion of data for training.")
    parser.add_argument("--val_split", default=0.2, type=float, help="Proportion of data for validation.")

    args = parser.parse_args()

    main(args)
