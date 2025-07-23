import os
import matplotlib.pyplot as plt

def save_plots(train_loss, val_dice, val_iou, val_interval, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure("Training Loss", (12, 6))
    plt.title("Epoch Average Loss")
    x_loss = [i + 1 for i in range(len(train_loss))]
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x_loss, train_loss, color='red', linewidth=2, label="train_loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "epoch_loss.png"))
    plt.close()

    plt.figure("Validation Metrics", (12, 6))
    plt.title("Validation Performance")
    x_val = [val_interval * (i + 1) for i in range(len(val_dice))]
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.plot(x_val, val_dice, color='blue', linewidth=2, label="Validation Mean Dice")
    plt.plot(x_val, val_iou, color='green', linewidth=2, label="Validation Mean IoU")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "validation_metrics.png"))
    plt.close()