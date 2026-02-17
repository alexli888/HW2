import numpy as np
import argparse

def generate_dataset(num_points, output_file):
    """
    Generate dataset of 2D integer points with x,y in range [0, 5000]
    """
    data = np.random.randint(0, 5001, size=(num_points, 2))
    np.savetxt(output_file, data, fmt="%d")
    print(f"Dataset with {num_points} integer points saved to {output_file}")


def generate_seeds(k, output_file):
    """
    Generate K integer seed points with x,y in range [0, 10000]
    """
    seeds = np.random.randint(0, 10001, size=(k, 2))
    np.savetxt(output_file, seeds, fmt="%d")
    print(f"{k} integer seed points saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset & K seed points")
    parser.add_argument("--num_points", type=int, default=3000,
                        help="Number of 2D data points (default: 3000)")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of seed points")
    
    args = parser.parse_args()

    generate_dataset(args.num_points, "dataset.txt")
    generate_seeds(args.k, "seeds.txt")
