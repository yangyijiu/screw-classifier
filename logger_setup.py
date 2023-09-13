from evaluation import *
import datetime
import os
import logging
from torch.utils.tensorboard import SummaryWriter
# Create a directory for the current run
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
run_dir = os.path.join("runs", current_time)
os.makedirs(run_dir, exist_ok=True)

# Set the filename for the log
log_filename = os.path.join(run_dir, "training.log")

# Configure logging
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# Set up TensorBoard writer
writer = SummaryWriter(log_dir=run_dir)

def log_metrics(epoch, total_loss, train_loader, model, val_loader,test_loader, writer):
    avg_loss = total_loss / len(train_loader)
    val_f1 = evaluate(model, val_loader)
    test_f1 = evaluate(model, test_loader)
    val_auc_roc = compute_auc_roc(model, val_loader)
    test_auc_roc = compute_auc_roc(model, test_loader)
    # Console output
    print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}"
          f", Validation F1 score: {val_f1:.4f}, Test F1 score: {test_f1:.4f}, Validation AUC-ROC: {val_auc_roc:.4f}, Test AUC-ROC: {test_auc_roc:.4f}")

    # TensorBoard logging
    writer.add_scalar('Training Loss', avg_loss, epoch)
    writer.add_scalar('Validation F1 Score', val_f1, epoch)
    writer.add_scalar('Test F1 Score', test_f1, epoch)
    writer.add_scalar('Validation AUC-ROC', val_auc_roc, epoch)
    writer.add_scalar('Test AUC-ROC', test_auc_roc, epoch)

    # File logging
    logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}'
                 f", Validation F1 score: {val_f1:.4f}, Test F1 score: {test_f1:.4f}, Validation AUC-ROC: {val_auc_roc:.4f}, Test AUC-ROC: {test_auc_roc:.4f}")
