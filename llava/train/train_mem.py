from llava.train.train import train
import wandb

# wandb.login(key="")

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
