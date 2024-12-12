import wandb
from sweep import train_jepa
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a W&B agent for a specific sweep')
    parser.add_argument('--sweep_id', type=str, required=True, help='The ID of the sweep to run')
    parser.add_argument('--project', type=str, default='wall_jepa_sweep', help='The name of the W&B project')
    parser.add_argument('--entity', type=str, default='gandhiji-u', help='The W&B entity (username or team name)')

    args = parser.parse_args()

    wandb.agent(args.sweep_id, train_jepa, project=args.project, entity=args.entity)
