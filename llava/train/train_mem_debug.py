from llava.train.train import train

import wandb

wandb.login(key="f3be6886613a8a311db063cba7d5e42fd396b9e9")


if __name__ == "__main__":
    breakpoint()
    train(attn_implementation="flash_attention_2")
