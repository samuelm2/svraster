import argparse
from src.sparse_voxel_model import SparseVoxelModel
from src.config import cfg, update_argparser, update_config
import os



def convert_to_ply(model_path, output_path):
    """
    Convert a .pt file to .ply format using SVInOut class.
    
    Args:
        input_path (str): Path to input .pt file
        output_path (str): Path where to save the .ply file
    """
    # Create SVInOut instance
        # Load config
    update_config(os.path.join(model_path, 'config.yaml'))
    model = SparseVoxelModel(cfg.model)
    
    # Load the .pt file
    print(f"Loading {model_path}...")
    model.load_iteration(-1)
    
    # Save as PLY
    print(f"Saving to {output_path}...")
    model.save_ply(output_path)
    
    print("Conversion complete!")   

def main():
    parser = argparse.ArgumentParser(description='Convert model to .ply format')
    parser.add_argument('model_path', type=str, help='Path to model output directory')
    parser.add_argument('output_path', type=str, help='Path where to save the .ply file')
    
    args = parser.parse_args()
    
    convert_to_ply(args.model_path, args.output_path)

if __name__ == "__main__":
    main()